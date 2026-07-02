#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=ou_sloan_gpu
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=hsdlp-miplib-small
#SBATCH --array=1-10%5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=botang@mit.edu
#SBATCH --output=/home/botang/src/cuHSDLP/slurm/slurm-%x-%A_%a.out

PROJ_DIR="/home/botang/src/cuHSDLP"
DATASET_DIR="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017"
INSTANCE_FILE="${PROJ_DIR}/scripts/miplib_instance.txt"

mkdir -p "${PROJ_DIR}/slurm"

TODAY="${SUBMIT_DATE:-$(date +%Y%m%d)}"
OUTPUT_DIR="${PROJ_DIR}/result/miplib_small_${TODAY}"
mkdir -p "$OUTPUT_DIR"

# load all Small instance names
mapfile -t all_instances < <(awk '$2 == "Small" {print $1}' "$INSTANCE_FILE" | head -100)

# split into chunks of 10: task 1 gets 1-10, task 2 gets 11-20, ...
CHUNK_SIZE=10
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE ))
chunk=("${all_instances[@]:$START:$CHUNK_SIZE}")

echo "Task ${SLURM_ARRAY_TASK_ID}: ${#chunk[@]} instances (index $((START+1))-$((START+${#chunk[@]})))"

for instance_name in "${chunk[@]}"; do
    instance_path="${DATASET_DIR}/${instance_name}.mps.gz"
    summary_file="${OUTPUT_DIR}/${instance_name}_summary.txt"

    echo "Instance: $instance_name"

    # skip if already solved
    if [ -f "$summary_file" ]; then
        echo "Skipping (already solved)"
        echo "--------------------------------------"
        continue
    fi

    # run solver (|| true prevents a crash from killing the whole task)
    "${PROJ_DIR}/build/cuhsdlp" \
        --time_limit 3600 \
        "$instance_path" "$OUTPUT_DIR" \
        || echo "WARNING: solver failed on $instance_name (exit code $?)"

    echo "Finished: $instance_name"
    echo "--------------------------------------"
done

echo "Task ${SLURM_ARRAY_TASK_ID} completed."
