#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:-/cluster/home/liaolw/dataset/3drealcar_zzj/car_orbit_quarters_shape_filter/car_orbit_quarters_top_aspect_ge_2p0_aesthetic_ge4p5_links/manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/cluster/home/liaolw/dataset/3drealcar_trellis}"
WORLD_SIZE="${WORLD_SIZE:-4}"
MODEL_NAME="${MODEL_NAME:-/cluster/home/liaolw/code/origin_TRELLIS/zzj_ckpts/TRELLIS-image-large}"

mkdir -p "${OUTPUT_DIR}/logs"
cd "${REPO_ROOT}"

pids=()
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    CUDA_VISIBLE_DEVICES="${rank}" python -u dataset_toolkits/export_vehicle_teacher_latents.py \
        --manifest "${MANIFEST}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_name "${MODEL_NAME}" \
        --rank "${rank}" \
        --world_size "${WORLD_SIZE}" \
        >"${OUTPUT_DIR}/logs/export_rank${rank}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if [[ "${status}" -ne 0 ]]; then
    echo "At least one export rank failed. Check ${OUTPUT_DIR}/logs and failures_rank*.jsonl." >&2
    exit "${status}"
fi

python dataset_toolkits/build_vehicle_distillation_dataset.py \
    --manifest "${MANIFEST}" \
    --teacher_dir "${OUTPUT_DIR}" \
    --validate headers \
    --condition_mode reference \
    --val_ratio 0.05

echo "Vehicle distillation dataset is ready at ${OUTPUT_DIR}."
