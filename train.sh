srun --partition=gpunodes --mem=8G --gres=gpu:rtx_a6000 \
    python tools/train.py configs/STFormer/stformer_base.py \
    --work_dir "./train_dir"