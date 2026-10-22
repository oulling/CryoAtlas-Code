#!/usr/bin/env bash
# CryoAtlas addition: portable CNN training/resume launcher.
# Checks independent manifests, initializes the architecture, and exposes
# process count, precision, batch size, and accumulation without private paths.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 TRAIN_LIST VAL_LIST RUN_DIR [CNN arguments...]" >&2
  exit 2
fi
train_list=$1
val_list=$2
run_dir=$3
shift 3
release_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$release_dir${PYTHONPATH:+:$PYTHONPATH}"
python "$release_dir/scripts/check_release.py" --train-list "$train_list" --val-list "$val_list"
mkdir -p "$run_dir"
if [[ ! -f "$run_dir/model.py" ]]; then
  cp "$release_dir/model_angelo/model_definitions/nucleotides/c_alpha/model.py" "$run_dir/model.py"
fi
distributed_args=()
if (( ${NUM_PROCESSES:-1} > 1 )); then
  distributed_args+=(--multi_gpu)
fi
exec accelerate launch "${distributed_args[@]}" --num_processes "${NUM_PROCESSES:-1}" \
  --num_machines 1 --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --dynamo_backend no -m model_angelo.c_alpha.main \
  --dataset-list "$train_list" --valid-dataset-list "$val_list" \
  --log-dir "$run_dir" --box-size 64 --batch-size "${BATCH_SIZE:-12}" \
  --gradient-accumulation-steps "${ACCUMULATION_STEPS:-2}" \
  --num-workers 8 --num-steps 40000 --scheduler-steps 135000 --seed 2026 \
  --lr 7.4e-4 --weight-decay 1e-2 --use-cosine-annealing \
  --tversky-warmup-steps 13500 --tversky-ramp-steps 40000 \
  --tversky-ramp-shape cosine --tversky-weight 1.0 \
  --random-crop-prob 0.25 --bfactor-range 25 --grad-clip 0.5 \
  --channel-weights "1.0,1.0,0.5" --focal-late-boost-start 68000 \
  --focal-late-boost-end 108000 --focal-late-boost-max 1.5 \
  --valid-every 2000 --valid-crops 200 --checkpoint-every 5000 \
  --log-every 20 --no-wandb "$@"
