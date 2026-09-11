# FastVideo Ascend 910B Wan2.2 causal SFT + DMD2 quality patch

Target upstream revision: `7bb76b5ec99807a66aa3047b901f15019abe0f00`.

This layered patch adds an accuracy-first, dense Ascend path for:

1. Wan2.2 TI2V-5B block-causal diffusion-forcing SFT.
2. Self-Forcing DMD2 quality recovery with a retained 50-step schedule.
3. DCP export and offline causal multi-step inference.

It can be installed on the fixed pristine FastVideo revision or on the existing
MiniMax-H3 Ascend patched tree.  It intentionally uses the repository's
old-style `sed -i` whole-file replacement installer.

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
