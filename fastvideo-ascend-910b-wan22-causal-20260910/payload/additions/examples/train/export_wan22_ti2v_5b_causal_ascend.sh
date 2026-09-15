#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <sft|dmd2> <checkpoint-or-output-dir> [export-dir]" >&2
  exit 2
fi

STAGE="$1"
CHECKPOINT="$2"
case "${STAGE}" in
  sft)
    DEFAULT_OUTPUT="outputs/wan22_ti2v_5b_causal_sft_export"
    PIPELINE_CLASS="WanCausalPipeline"
    ;;
  dmd2)
    DEFAULT_OUTPUT="outputs/SFWan2.2-TI2V-5B-Ascend-Causal"
    PIPELINE_CLASS="WanCausalDMDPipeline"
    ;;
  *)
    echo "Unknown stage: ${STAGE}; expected sft or dmd2" >&2
    exit 2
    ;;
esac
OUTPUT="${3:-${DEFAULT_OUTPUT}}"

python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "${CHECKPOINT}" \
  --role student \
  --output-dir "${OUTPUT}" \
  --overwrite \
  --verify

python - "${OUTPUT}" "${PIPELINE_CLASS}" "${CHECKPOINT}" "${STAGE}" <<'PY'
import glob
import json
import os
import sys

root = os.path.abspath(sys.argv[1])
pipeline_class = sys.argv[2]
checkpoint = os.path.abspath(sys.argv[3])
stage = sys.argv[4]
updates = [
    (os.path.join(root, "model_index.json"), "_class_name", pipeline_class),
    (os.path.join(root, "transformer", "config.json"), "_class_name", "CausalWanTransformer3DModel"),
]
for path, key, value in updates:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data[key] = value
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)

causal_config = {
    "pipeline_class": pipeline_class,
    "dmd_denoising_steps": None,
    "warp_denoising_step": None,
}
if stage == "dmd2":
    metadata_paths = [os.path.join(checkpoint, "metadata.json")]
    metadata_paths.extend(glob.glob(os.path.join(checkpoint, "checkpoint-*", "metadata.json")))
    metadata_paths = [path for path in metadata_paths if os.path.isfile(path)]
    if not metadata_paths:
        raise FileNotFoundError(
            f"Cannot record the DMD2 inference schedule: no metadata.json under {checkpoint}"
        )
    metadata_path = max(metadata_paths, key=os.path.getmtime)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle).get("config", {})
    method_config = raw_config.get("method", {})
    steps = method_config.get("dmd_denoising_steps")
    if not steps:
        raise ValueError(f"Missing method.dmd_denoising_steps in {metadata_path}")
    causal_config["dmd_denoising_steps"] = [int(step) for step in steps]
    causal_config["warp_denoising_step"] = bool(
        method_config.get("warp_denoising_step", False)
    )

causal_config_path = os.path.join(root, "fastvideo_causal_config.json")
temporary = causal_config_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(causal_config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(temporary, causal_config_path)
print(f"Stamped {pipeline_class} metadata in {root}")
PY

echo "Wan2.2 causal ${STAGE} export complete: ${OUTPUT}"
