import argparse

from fastvideo import SamplingParam, VideoGenerator
from fastvideo.configs.pipelines.wan import Wan2_2_TI2V_5B_Config


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
    args = parser.parse_args()

    pipeline_config = Wan2_2_TI2V_5B_Config()
    pipeline_config.is_causal = True
    pipeline_config.dmd_denoising_steps = list(range(1000, 0, -20))
    pipeline_config.warp_denoising_step = True

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        pipeline_config=pipeline_config,
        override_pipeline_cls_name="WanCausalDMDPipeline",
        override_transformer_cls_name="CausalWanTransformer3DModel",
        num_gpus=args.num_gpus,
        use_fsdp_inference=args.num_gpus > 1,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        num_frame_per_block=3,
        dmd_denoising_steps=list(range(1000, 0, -20)),
    )
    # Use the local export as the preset lookup key.  This keeps inference
    # fully offline on clusters that cannot reach Hugging Face.
    sampling = SamplingParam.from_pretrained(args.model_path)
    sampling.height = args.height
    sampling.width = args.width
    sampling.num_frames = args.num_frames
    sampling.fps = args.fps
    sampling.seed = args.seed
    sampling.num_inference_steps = 50
    sampling.guidance_scale = 1.0
    generator.generate_video(
        args.prompt,
        sampling_param=sampling,
        output_path=args.output,
        save_video=True,
    )


if __name__ == "__main__":
    main()
