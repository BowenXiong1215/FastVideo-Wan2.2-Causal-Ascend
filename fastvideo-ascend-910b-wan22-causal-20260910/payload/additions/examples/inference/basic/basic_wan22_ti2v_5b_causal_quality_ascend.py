import argparse
import json
import os

from fastvideo import SamplingParam, VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_TI2V_5B_Config


def _parse_steps(value: str) -> list[int]:
    try:
        steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "denoising steps must be comma-separated integers"
        ) from exc
    if not steps or any(step <= 0 or step > 1000 for step in steps):
        raise argparse.ArgumentTypeError(
            "denoising steps must contain integers in the range 1..1000"
        )
    return steps


def _exported_dmd_steps(model_path: str) -> list[int] | None:
    config_path = os.path.join(model_path, "fastvideo_causal_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_steps = json.load(handle).get("dmd_denoising_steps")
    if raw_steps is None:
        return None
    return _parse_steps(",".join(str(step) for step in raw_steps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="outputs/wan22_causal_quality")
    parser.add_argument("--prompt", default="A cinematic view of a car driving through a mountain road at sunset.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--stage", choices=("auto", "sft", "dmd2"), default="auto")
    parser.add_argument(
        "--dmd-denoising-steps",
        type=_parse_steps,
        help=("override the DMD2 schedule, for example 1000,750,500,250; "
              "by default it is read from the export metadata"),
    )
    args = parser.parse_args()

    pipeline_config = Wan2_2_TI2V_5B_Config()
    pipeline_config.is_causal = True

    if args.stage == "auto":
        model_index_path = os.path.join(args.model_path, "model_index.json")
        with open(model_index_path, "r", encoding="utf-8") as handle:
            exported_pipeline_class = json.load(handle).get("_class_name")
        if exported_pipeline_class == "WanCausalPipeline":
            stage = "sft"
        elif exported_pipeline_class == "WanCausalDMDPipeline":
            stage = "dmd2"
        else:
            raise ValueError(
                f"Unsupported causal Wan pipeline in {model_index_path}: "
                f"{exported_pipeline_class!r}"
            )
    else:
        stage = args.stage

    dmd_steps = None
    if stage == "sft":
        pipeline_class = "WanCausalPipeline"
    else:
        pipeline_class = "WanCausalDMDPipeline"
        dmd_steps = args.dmd_denoising_steps or _exported_dmd_steps(args.model_path)
        if dmd_steps is None:
            raise ValueError(
                "DMD2 schedule metadata is missing. Re-export the checkpoint or "
                "pass --dmd-denoising-steps explicitly."
            )
        pipeline_config.dmd_denoising_steps = dmd_steps
        pipeline_config.warp_denoising_step = True

    print(f"Using {stage} inference pipeline: {pipeline_class}; steps={dmd_steps}")

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        pipeline_config=pipeline_config,
        override_pipeline_cls_name=pipeline_class,
        override_transformer_cls_name="CausalWanTransformer3DModel",
        num_gpus=args.num_gpus,
        # Keep causal KV-cache tokens replicated. FastVideo otherwise defaults
        # sp_size to num_gpus, which makes RoPE 8x longer than the local token
        # sequence in an 8-device FSDP inference job.
        tp_size=1,
        sp_size=1,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=args.num_gpus,
        use_fsdp_inference=args.num_gpus > 1,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        num_frame_per_block=3,
        dmd_denoising_steps=dmd_steps,
    )
    # Use the local export as the preset lookup key.  This keeps inference
    # fully offline on clusters that cannot reach Hugging Face.
    sampling = SamplingParam.from_pretrained(args.model_path)
    sampling.height = args.height
    sampling.width = args.width
    sampling.num_frames = args.num_frames
    sampling.fps = args.fps
    sampling.seed = args.seed
    sampling.num_inference_steps = len(dmd_steps) if dmd_steps is not None else 50
    sampling.guidance_scale = 1.0
    generator.generate_video(
        args.prompt,
        sampling_param=sampling,
        output_path=args.output,
        save_video=True,
    )


if __name__ == "__main__":
    main()
