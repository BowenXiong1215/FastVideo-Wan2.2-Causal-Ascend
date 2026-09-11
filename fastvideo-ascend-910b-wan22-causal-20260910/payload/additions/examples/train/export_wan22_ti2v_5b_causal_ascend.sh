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
    CONFIG="examples/train/configs/ascend/wan2_2_ti2v_5b_causal_sft.yaml"
    DEFAULT_OUTPUT="outputs/wan22_ti2v_5b_causal_sft_export"
    PIPELINE_CLASS="WanCausalPipeline"
    ;;
  dmd2)
    CONFIG="examples/train/configs/ascend/wan2_2_ti2v_5b_causal_dmd2_quality.yaml"
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
  --config "${CONFIG}" \
  --role student \
  --output-dir "${OUTPUT}" \
  --overwrite \
  --verify

python - "${OUTPUT}" "${PIPELINE_CLASS}" <<'PY'
import json
import os
import sys

root = os.path.abspath(sys.argv[1])
pipeline_class = sys.argv[2]
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
print(f"Stamped {pipeline_class} metadata in {root}")
PY

echo "Wan2.2 causal ${STAGE} export complete: ${OUTPUT}"
