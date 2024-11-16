#!/bin/bash

#SBATCH --partition=gpunodes        # Specify the GPU partition
#SBATCH --mem=8G                    # Memory allocation
#SBATCH --nodelist=peleus            # Node to use
#SBATCH --gres=gpu:2   # Request one RTX 6000 Ada GPU
#SBATCH --time=23:00:00


# Print GPU info to verify allocation
echo "Selected GPU ID(s): ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"

torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    --nnodes=1 \
    --nproc_per_node=1 \
    tools/train.py configs/STFormer/stformer_base.py \
    --distributed=True --work_dir "./train_dir" \
    --resume "./train_dir/checkpoints/epoch_12.pth"

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=localhost:0 \
#     --nnodes=1 \
#     --nproc_per_node=1 \
#     tools/train.py configs/STFormer/stformer_base_4.py \
#     --distributed=True --work_dir "./train_dir_4" 