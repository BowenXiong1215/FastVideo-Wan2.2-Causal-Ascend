# Wan2.2 TI2V-5B causal SFT + DMD2 quality recovery on Ascend 910B

This path turns the released bidirectional Wan2.2 TI2V-5B transformer into a
block-causal student, then uses FastVideo's Self-Forcing DMD2 method to recover
quality under the student's own autoregressive history. The recommended bring-up
recipe distills to four denoising steps and uses stochastic gradient truncation:
one denoising position is sampled per sequence, only that position retains a
gradient. On student iterations, the final one-frame remainder block is the
only block retaining an autograd graph; previous causal frames/KV remain
detached. This gives a deterministic graph-memory bound on 910B. Causal
training uses exact PyTorch SDPA rather than VSA or an approximate sparse mask.

This is an engineering bring-up configuration, not a published Wan2.2-5B
quality recipe. Completing the commands proves that the two-stage training,
export and causal multi-step inference paths work on 910B. Dataset scale,
learning rates and guidance still need task-specific quality tuning.

Run every command below from the FastVideo source root.

## 1. Inputs

Use a local Diffusers-format copy of `Wan-AI/Wan2.2-TI2V-5B-Diffusers`.  The
cluster does not need Hugging Face access.  Check the model and, once created,
the preprocessed dataset with:

```bash
python scripts/verify_wan22_causal_ascend.py /models/Wan2.2-TI2V-5B-Diffusers
```

Prepare videos and captions in this layout:

```text
/data/wan22_raw/
├── merge.txt
├── videos/
│   └── clip_001.mp4
└── annotation.json
```

`merge.txt` contains:

```text
videos,annotation.json
```

`annotation.json` is a JSON list.  Each item contains a path relative to the
video directory and at least one caption:

```json
[
  {
    "path": "clip_001.mp4",
    "cap": ["A car follows a winding mountain road at sunset."],
    "resolution": {"width": 1280, "height": 704},
    "fps": 24.0,
    "duration": 5.1,
    "num_frames": 121
  }
]
```

Preprocess on one NPU (the current FastVideo preprocessor requires world size
one):

```bash
bash examples/train/preprocess_wan22_ti2v_5b_ascend.sh \
  /models/Wan2.2-TI2V-5B-Diffusers \
  /data/wan22_raw/merge.txt \
  /data/wan22_processed
```

The training data path is the generated
`/data/wan22_processed/combined_parquet_dataset` directory.

Verify the local model and the generated parquet dataset together:

```bash
python scripts/verify_wan22_causal_ascend.py \
  /models/Wan2.2-TI2V-5B-Diffusers \
  --data-path /data/wan22_processed/combined_parquet_dataset
```

## 2. Stage A: block-causal SFT

First run configuration validation without loading model weights:

```bash
bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_sft.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --dry-run
```

Then do a one-step hardware smoke test:

```bash
NUM_GPUS=8 WANDB_MODE=offline bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_sft.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --training.loop.max_train_steps 1 \
  --training.checkpoint.training_state_checkpointing_steps 1 \
  --training.checkpoint.output_dir /outputs/wan22_sft_smoke
```

Run the full SFT job with a separate output directory:

```bash
NUM_GPUS=8 WANDB_MODE=offline bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_sft.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --training.checkpoint.output_dir /outputs/wan22_sft
```

Important SFT parameters:

- `chunk_size` and `num_frames_per_block` must match.  Three latent frames is
  the default causal block.
- `min_timestep_ratio` / `max_timestep_ratio` select the noise range used by
  diffusion-forcing SFT.
- `learning_rate` controls the causal student's update size.
- `max_train_steps` is a bring-up default, not a quality guarantee.
- `TORCH_SDPA` selects the exact dense NPU-compatible attention path.

Resume an interrupted run with:

```bash
NUM_GPUS=8 WANDB_MODE=offline bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_sft.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --training.checkpoint.resume_from_checkpoint /outputs/wan22_sft/checkpoint-500
```

Export the chosen SFT checkpoint.  Export is a single-process job and may take
several minutes because it gathers the distributed 5B state dict on CPU:

```bash
bash examples/train/export_wan22_ti2v_5b_causal_ascend.sh \
  sft \
  /outputs/wan22_sft/checkpoint-4000 \
  /outputs/wan22_sft_export
```

## 3. Stage B: Self-Forcing DMD2 quality recovery

The second stage uses three model roles: the exported causal student, a frozen
bidirectional teacher, and a trainable bidirectional critic. The student uses
the four-point `[1000, 750, 500, 250]` schedule and generates block by block.
Teacher and critic scores correct the distribution drift caused by the causal
student consuming its own generated history.

```bash
bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_dmd2_4step.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.student.transformer_override_safetensor /outputs/wan22_sft_export/transformer/model.safetensors \
  --models.teacher.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.critic.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --dry-run
```

Use a two-step hardware smoke test with an isolated output directory. The
temporary `generator_update_interval: 2` override makes iteration one exercise
the critic and iteration two exercise the student, without holding their
autograd graphs at the same time:

```bash
NUM_GPUS=8 WANDB_MODE=offline bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_dmd2_4step.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.student.transformer_override_safetensor /outputs/wan22_sft_export/transformer/model.safetensors \
  --models.teacher.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.critic.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --method.generator_update_interval 2 \
  --training.loop.max_train_steps 2 \
  --training.checkpoint.training_state_checkpointing_steps 1 \
  --training.checkpoint.output_dir /outputs/wan22_dmd2_smoke
```

Then run the 49-frame recovery probe. It saves checkpoints at steps 25, 50, 75
and 100 so quality and stability can be compared before extending the run:

```bash
NUM_GPUS=8 WANDB_MODE=offline bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_dmd2_4step.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.student.transformer_override_safetensor /outputs/wan22_sft_export/transformer/model.safetensors \
  --models.teacher.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.critic.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --training.checkpoint.output_dir /outputs/wan22_dmd2_4step_49f
```

After 49 frames is stable, use this 81-frame tier. `21` latent frames maps to
`81` RGB frames for Wan's temporal VAE:

```bash
NUM_GPUS=8 WANDB_MODE=offline bash examples/train/run.sh \
  examples/train/configs/ascend/wan2_2_ti2v_5b_causal_dmd2_4step.yaml \
  --models.student.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.student.transformer_override_safetensor /outputs/wan22_sft_export/transformer/model.safetensors \
  --models.teacher.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --models.critic.init_from /models/Wan2.2-TI2V-5B-Diffusers \
  --training.data.data_path /data/wan22_processed/combined_parquet_dataset \
  --training.data.num_latent_t 21 \
  --training.data.num_frames 81 \
  --training.checkpoint.output_dir /outputs/wan22_dmd2_4step_81f
```

Important DMD2 parameters:

- `dmd_denoising_steps: [1000, 750, 500, 250]` is the four-step deployment
  schedule as well as the on-policy rollout schedule.
- `generator_update_interval` alternates critic and student work; `5` means
  one student update every five iterations.
- `real_score_guidance_scale` controls teacher CFG during the DMD target.
- `fake_score_learning_rate` trains the critic; `training.optimizer.learning_rate`
  trains the causal student.
- `student_sample_type: ode` keeps the causal rollout deterministic.
- `last_step_only: false` and `same_step_across_blocks: true` sample one of the
  four denoising positions per sequence. Earlier positions run under
  `no_grad`; previous generated frames and KV are detached. Across iterations,
  every one of the four positions receives supervision.
- `gradient_block_mode: last_one` retains only the final one-frame remainder
  from the `3+3+3+3+1` latent layout. The loss is rescaled from a sequence mean
  to that frame's mean. Unlike `random_one`, its memory ceiling cannot jump
  threefold when a regular three-frame block is selected.
- `generator_update_interval: 5` means iterations 1-4 train only the critic and
  iteration 5 trains only the student. This is the OOM-safe alternating path.

The older `wan2_2_ti2v_5b_causal_dmd2_quality.yaml` remains available for an
explicit 50-step ablation, but it is not the recommended first recovery run.

Export the selected DMD2 checkpoint:

```bash
bash examples/train/export_wan22_ti2v_5b_causal_ascend.sh \
  dmd2 \
  /outputs/wan22_dmd2_4step_49f/checkpoint-100 \
  /outputs/SFWan2.2-TI2V-5B-Ascend-Causal
```

The export helper constructs only the requested student role and asks DCP for
only `roles.student.transformer`. It does not instantiate the 5B teacher,
critic, or either optimizer on the single export device. Before strict reload
verification it releases the exported student and clears the accelerator
allocator.

## 4. Causal multi-step inference

```bash
python examples/inference/basic/basic_wan22_ti2v_5b_causal_quality_ascend.py \
  --model-path /outputs/SFWan2.2-TI2V-5B-Ascend-Causal \
  --output /outputs/wan22_causal \
  --prompt "A car follows a winding mountain road at sunset." \
  --num-gpus 8
```

For a direct comparison with the short training probe, pass `--num-frames 49`.
After the 81-frame tier, pass `--num-frames 81`. Wan2.2 uses 13 and 21 latent
frames respectively, and the causal runtime accepts the final one-frame
remainder after regular three-frame blocks.

The export records the checkpoint's DMD schedule in
`fastvideo_causal_config.json`; inference reads it automatically. For an older
export without that file, either re-export it or pass
`--dmd-denoising-steps 1000,750,500,250` explicitly.

This implementation establishes a causal text-to-video backbone.  It does not
add action, keyboard, camera-pose, or robot-control conditioning, so by itself
it is not yet a controllable interactive world model.
