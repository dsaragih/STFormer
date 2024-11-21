#!/bin/bash

# srun --partition=debugnodes --mem=8G --nodelist=tensor1 --gres=gpu:nvidia_titan_v \
#     python tools/test.py configs/STFormer/stformer_base.py --weights=train_dir/checkpoints/epoch_45.pth

# srun --partition=debugnodes --mem=8G --nodelist=tensor1 --gres=gpu:nvidia_titan_v \
#     python tools/train.py configs/STFormer/stformer_base_test.py --resume=train_dir/checkpoints/epoch_45.pth --work_dir "./work_eval"
srun --partition=debugnodes --mem=8G --nodelist=tensor1 --gres=gpu:nvidia_titan_v \
    python tools/train.py configs/STFormer/stformer_base_6.py \
         --work_dir "./train_dir_6" 
