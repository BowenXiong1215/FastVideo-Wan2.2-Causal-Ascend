#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export one model role from a FastVideo DCP checkpoint.

Unlike the generic training-method exporter, this path never constructs the
other DMD2 roles or their optimizers. This is important for 5B student exports
on a single 64-GB accelerator.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any


def _release_accelerator_memory() -> None:
    import torch

    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def export_role(
    *,
    checkpoint: str,
    output_dir: str,
    role: str,
    overwrite: bool,
    verify: bool,
) -> str:
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.train.entrypoint.dcp_to_diffusers import (
        _ensure_distributed,
        _run_config_from_raw,
        _save_role_pretrained,
        _strict_reload_verify,
    )
    from fastvideo.train.utils.checkpoint import (
        CheckpointManager,
        _resolve_resume_checkpoint,
    )
    from fastvideo.train.utils.instantiate import instantiate
    from fastvideo.training.checkpointing_utils import ModelWrapper

    import torch.distributed.checkpoint as dcp

    _ensure_distributed()
    resolved = _resolve_resume_checkpoint(checkpoint, output_dir=checkpoint)
    dcp_dir = resolved / "dcp"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(f"Missing dcp/ under {resolved}")

    metadata = CheckpointManager.load_metadata(resolved)
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"Missing checkpoint config in {resolved / 'metadata.json'}")
    cfg = _run_config_from_raw(raw_config)
    if role not in cfg.models:
        raise KeyError(f"Role {role!r} is not present in checkpoint config")

    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    tc.distributed.tp_size = 1
    tc.distributed.sp_size = 1
    tc.distributed.num_gpus = 1
    tc.distributed.hsdp_replicate_dim = 1
    tc.distributed.hsdp_shard_dim = 1

    # Build only the requested architecture. DCP overwrites every transformer
    # parameter, so loading the SFT override and enabling training/checkpointing
    # would only consume extra memory.
    model_config: dict[str, Any] = dict(cfg.models[role])
    model_config["trainable"] = False
    model_config.pop("transformer_override_safetensor", None)
    model_config.pop("enable_gradient_checkpointing_type", None)
    model = instantiate(model_config, training_config=tc)
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise ValueError(f"Role {role!r} does not contain a transformer")

    states = {f"roles.{role}.transformer": ModelWrapper(transformer)}
    dcp.load(states, checkpoint_id=str(dcp_dir))

    base_model_path = str(model_config.get("init_from", ""))
    if not base_model_path:
        raise ValueError(f"models.{role}.init_from is missing")
    result = _save_role_pretrained(
        role=role,
        base_model_path=base_model_path,
        output_dir=output_dir,
        overwrite=overwrite,
        model=model,
    )

    if verify:
        del states
        del transformer
        del model
        _release_accelerator_memory()
        _strict_reload_verify(output_dir=result, training_config=tc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--role", default="student")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    result = export_role(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        role=args.role,
        overwrite=args.overwrite,
        verify=args.verify,
    )
    print(f"Role-only export complete: {Path(result).resolve()}")


if __name__ == "__main__":
    main()
