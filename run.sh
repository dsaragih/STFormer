srun --partition=gpunodes --mem=8G --gres=gpu:rtx_a6000 \
    python tools/test.py configs/STFormer/stformer_base.py --weights=checkpoints/stformer_base.pth