#!/bin/bash

srun --partition=debugnodes --mem=8G --nodelist=tensor1 --gres=gpu:nvidia_titan_v \
    python tools/test.py configs/STFormer/stformer_base.py --weights=train_dir/checkpoints/epoch_8.pth