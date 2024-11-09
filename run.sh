#!/bin/bash

srun --partition=gpunodes --mem=8G --nodelist=peleus --gres=gpu:nvidia_titan_rtx \
    python tools/test.py configs/STFormer/stformer_base.py --weights=train_dir/checkpoints/epoch_7.pth