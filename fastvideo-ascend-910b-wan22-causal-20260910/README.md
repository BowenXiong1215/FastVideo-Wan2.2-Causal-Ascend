# FastVideo Ascend 910B Wan2.2 causal SFT + DMD2 quality patch

Target upstream revision: `7bb76b5ec99807a66aa3047b901f15019abe0f00`.

This layered patch adds an accuracy-first, dense Ascend path for:

1. Wan2.2 TI2V-5B block-causal diffusion-forcing SFT.
2. Four-step Self-Forcing DMD2 quality recovery with paper-style stochastic
   gradient truncation; the previous 50-step recipe remains as an ablation.
3. DCP export and offline causal multi-step inference.
4. NPU-aware pinned-memory routing for the Parquet StatefulDataLoader.
5. Checkpoint-metadata-driven export that preserves offline local model paths.

It can be installed on the fixed pristine FastVideo revision or on the existing
MiniMax-H3 Ascend patched tree.  It intentionally uses the repository's
old-style `sed -i` whole-file replacement installer.
Known older versions of files added by this bundle are upgraded by checksum;
unknown local modifications remain protected from overwrite.

DMD2 uses mutually exclusive critic and student iterations.  With
`generator_update_interval: 5`, four critic updates are followed by one student
update, avoiding simultaneous critic/student autograd graphs at the student
memory peak while preserving both objectives.

The recommended DMD2 config starts with 49 frames and stops at 100 iterations,
saving steps 25/50/75/100. An 81-frame tier is documented after the short run is
stable. Exported DMD checkpoints carry their exact inference schedule so a
four-step checkpoint cannot silently fall back to the old 50-step sampler.
Student iterations retain an autograd graph only for the final one-frame
remainder block and rescale the loss accordingly. This makes the graph-memory
ceiling deterministic instead of occasionally sampling a three-frame block.
The DCP exporter is role-only: a student export does not instantiate the 5B
teacher, critic, or optimizer states on the single export device.

```bash
tar -xzf fastvideo-ascend-910b-wan22-causal-20260910.tar.gz
bash fastvideo-ascend-910b-wan22-causal-20260910/install.sh /path/to/FastVideo
bash fastvideo-ascend-910b-wan22-causal-20260910/verify.sh /path/to/FastVideo
```

After installation, follow:

```text
docs/getting_started/ascend_910b_wan22_causal_distillation.md
```

The code and configuration are statically validated on the packaging host.
Actual training and numerical parity still require an Ascend 910B runtime.
