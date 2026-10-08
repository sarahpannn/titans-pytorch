#!/usr/bin/env bash
# Usage: scripts/finetune.sh gold|quality INIT_CHECKPOINT OUTPUT_DIR [extra SFT flags]
set -euo pipefail
cd "$(dirname "$0")/.."
stage=${1:?stage: gold or quality}
checkpoint=${2:?path to input latest_checkpoint.pt}
output=${3:?output directory}
shift 3
[ -s "$checkpoint" ] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
[ -s "$(dirname "$checkpoint")/training_config.json" ] || { echo "Missing colocated training_config.json" >&2; exit 1; }
common=(--ddp --init_checkpoint "$checkpoint" --output_dir "$output" --seqlen 16384 --batch_size 128 --micro_batch_size 2 --nmlr 1e-4 --warmup 5)
case "$stage" in
  gold) extras=(--dataset longalign_loongrl --longalign 9984 --loongrl 2496 --steps 200 --save_every 25) ;;
  quality) extras=(--dataset quality_mc --epochs 3 --save_every 4) ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node=2 sft_kvwrite_quality_mc.py "${common[@]}" "${extras[@]}" "$@"
