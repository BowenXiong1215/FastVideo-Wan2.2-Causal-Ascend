#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <local-model-path> <merge.txt> <output-dir>" >&2
  exit 2
fi

MODEL_PATH="$(cd "$1" && pwd)"
MERGE_PATH="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
OUTPUT_DIR="$3"

test -f "${MODEL_PATH}/model_index.json" || {
  echo "Missing ${MODEL_PATH}/model_index.json" >&2
  exit 3
}
test -f "${MERGE_PATH}" || {
  echo "Missing dataset manifest: ${MERGE_PATH}" >&2
  exit 3
}
mkdir -p "${OUTPUT_DIR}"

export TOKENIZERS_PARALLELISM=false

# FastVideo's v1 preprocessor currently enforces WORLD_SIZE=1.
python -m torch.distributed.run --nproc_per_node=1 \
  fastvideo/pipelines/preprocess/v1_preprocess.py \
  --model_path "${MODEL_PATH}" \
  --data_merge_path "${MERGE_PATH}" \
  --preprocess_video_batch_size 1 \
  --seed 42 \
  --max_height 704 \
  --max_width 1280 \
  --num_frames 121 \
  --dataloader_num_workers 0 \
  --output_dir "${OUTPUT_DIR}" \
  --train_fps 24 \
  --samples_per_file 8 \
  --flush_frequency 8 \
  --video_length_tolerance_range 5 \
  --preprocess_task t2v

echo "Wan2.2 TI2V-5B preprocessing complete: ${OUTPUT_DIR}"
