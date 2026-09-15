#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Statically verify a local Wan2.2 causal SFT + DMD2 setup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "examples/train/configs/ascend/wan2_2_ti2v_5b_causal_sft.yaml",
    ROOT / "examples/train/configs/ascend/wan2_2_ti2v_5b_causal_dmd2_quality.yaml",
    ROOT / "examples/train/configs/ascend/wan2_2_ti2v_5b_causal_dmd2_4step.yaml",
)


def _load_json(path: Path, errors: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"invalid JSON {path}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path}")
        return {}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--data-path", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    model = args.model_path.expanduser().resolve()
    if not model.is_dir():
        errors.append(f"model directory does not exist: {model}")
    else:
        index = _load_json(model / "model_index.json", errors)
        for component in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
            if not (model / component).is_dir():
                errors.append(f"missing model component: {component}/")
            if index and component not in index:
                warnings.append(f"model_index.json does not list {component!r}")
        transformer_config = _load_json(model / "transformer/config.json", errors)
        in_channels = transformer_config.get("in_channels")
        if in_channels is not None and int(in_channels) != 48:
            warnings.append(f"transformer in_channels={in_channels}; released Wan2.2 TI2V-5B normally uses 48")
        tensors = list(model.rglob("*.safetensors"))
        if not tensors:
            errors.append("no safetensors files found under model directory")
        for tensor in tensors:
            if tensor.stat().st_size == 0:
                errors.append(f"empty weight file: {tensor.relative_to(model)}")

    if args.data_path is not None:
        data = args.data_path.expanduser().resolve()
        if not data.is_dir():
            errors.append(f"preprocessed data directory does not exist: {data}")
        elif not any(data.rglob("*.parquet")):
            errors.append(f"no parquet files found under: {data}")

    try:
        import yaml
    except ImportError:
        warnings.append("PyYAML unavailable; skipped YAML parsing")
    else:
        for config in CONFIGS:
            try:
                parsed = yaml.safe_load(config.read_text(encoding="utf-8"))
            except Exception as error:
                errors.append(f"invalid YAML {config}: {error}")
                continue
            backend_names = []
            for role in parsed.get("models", {}).values():
                backend_names.append(role.get("attention_backend"))
            if any(name != "TORCH_SDPA" for name in backend_names):
                errors.append(f"all roles must select TORCH_SDPA: {config}")
            if config.name.endswith("causal_dmd2_quality.yaml"):
                method = parsed.get("method", {})
                steps = method.get("dmd_denoising_steps", [])
                if len(steps) != 50:
                    errors.append(f"quality-recovery DMD2 must retain 50 steps: {config}")
                if method.get("last_step_only") is not False:
                    errors.append(
                        f"quality-recovery DMD2 must use last_step_only=false: {config}"
                    )
                if method.get("student_sample_type") != "ode":
                    errors.append(f"quality-recovery DMD2 must use an ODE rollout: {config}")
            if config.name.endswith("causal_dmd2_4step.yaml"):
                method = parsed.get("method", {})
                training = parsed.get("training", {})
                data = training.get("data", {})
                loop = training.get("loop", {})
                checkpoint = training.get("checkpoint", {})
                if method.get("dmd_denoising_steps") != [1000, 750, 500, 250]:
                    errors.append(f"four-step DMD2 schedule is invalid: {config}")
                if method.get("last_step_only") is not False:
                    errors.append(f"four-step DMD2 must use stochastic truncation: {config}")
                if method.get("same_step_across_blocks") is not True:
                    errors.append(f"four-step DMD2 must sample one step per sequence: {config}")
                if data.get("num_latent_t") != 13 or data.get("num_frames") != 49:
                    errors.append(f"four-step bring-up must start at 49 frames: {config}")
                if loop.get("max_train_steps") != 100:
                    errors.append(f"four-step bring-up must stop at 100 iterations: {config}")
                if checkpoint.get("training_state_checkpointing_steps") != 25:
                    errors.append(f"four-step bring-up must checkpoint every 25 iterations: {config}")

    causal_source = (ROOT / "fastvideo/models/dits/causal_wanvideo.py").read_text(encoding="utf-8")
    if "_forward_blockwise_sdpa" not in causal_source:
        errors.append("causal Wan exact blockwise SDPA fallback is not installed")
    for relative in (
        "fastvideo/train/methods/distribution_matching/dmd2.py",
        "fastvideo/train/methods/distribution_matching/self_forcing.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        if 'attn_kind="vsa"' in source:
            errors.append(f"hard-coded VSA attention remains in {relative}")

    self_forcing_source = (
        ROOT / "fastvideo/train/methods/distribution_matching/self_forcing.py"
    ).read_text(encoding="utf-8")
    for marker in ("_sample_exit_indices", "torch.no_grad()", "store_kv=False"):
        if marker not in self_forcing_source:
            errors.append(f"stochastic gradient truncation marker is missing: {marker}")

    print(f"Model path: {model}")
    if args.data_path is not None:
        print(f"Data path:  {args.data_path.expanduser().resolve()}")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"Wan2.2 causal Ascend verification: FAIL ({len(errors)} error(s))", file=sys.stderr)
        return 1
    print("Wan2.2 causal Ascend verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
