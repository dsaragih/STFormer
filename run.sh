#!/bin/bash

srun --partition=gpunodes --mem=16G --nodelist=peleus --gres=gpu:1 \
    python tools/test.py configs/STFormer/stformer_base_test.py --weights=train_dir_8/checkpoints/epoch_15.pth --work_dir "work_eval/mask_8_epoch_15"
