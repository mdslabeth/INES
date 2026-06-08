#! /usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./submit_train_jobs.sh --data-dir DATA_DIR [options]

Options:
  --data-dir PATH        Directory containing forecasting input files. Required.
  --output-dir PATH      Output directory for logs and model artifacts.
                         Default: logs/final_forecasters
  --epochs N             Training epochs. Default: 64
  --batch-size N         Training batch size. Default: 2048
  --python-bin PATH      Python executable. Default: python
  -h, --help             Show this help text.
EOF
}

require_value() {
    if [[ $# -lt 2 || "${2}" == --* ]]; then
        echo "Missing value for ${1}" >&2
        exit 1
    fi
}

DATA_DIR=""
OUTPUT_DIR="logs/final_forecasters"
NUM_WORKERS=""
EPOCHS="64"
BATCH_SIZE="2048"
ACCELERATOR="auto"
PYTHON_BIN="python"
feature_sets=()

while [[ $# -gt 0 ]]; do
    case "${1}" in
        --data-dir)
            require_value "${1}" "${2:-}"
            DATA_DIR="${2}"
            shift 2
            ;;
        --output-dir)
            require_value "${1}" "${2:-}"
            OUTPUT_DIR="${2}"
            shift 2
            ;;
        --epochs)
            require_value "${1}" "${2:-}"
            EPOCHS="${2}"
            shift 2
            ;;
        --batch-size)
            require_value "${1}" "${2:-}"
            BATCH_SIZE="${2}"
            shift 2
            ;;
        --python-bin)
            require_value "${1}" "${2:-}"
            PYTHON_BIN="${2}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: ${1}" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${DATA_DIR}" ]]; then
    echo "--data-dir must point to the directory containing forecasting input files." >&2
    usage >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/slurm_logs"
cpus=8

job_name=all_combis
sbatch \
    -o "${OUTPUT_DIR}/slurm_logs/${job_name}.log" \
    -J "${job_name}" \
    -c "${cpus}" \
    -t "00:30:00" \
    -p gpu \
    --gres=gpu:1 \
    --mem=16G \
    --wrap \
    "${PYTHON_BIN} train_forecaster.py \
    --data_dir \"${DATA_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --epochs \"${EPOCHS}\" \
    --batch_size \"${BATCH_SIZE}\" \
    --num_workers \"${cpus}\" \
    --accelerator "auto" "