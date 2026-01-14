#!/bin/bash
# 训练脚本

# 设置模型路径（本地路径）
MODEL_PATH="/home/ruqi/public/Qwen3-8B"

# 设置数据路径
TRAIN_DATA_DIR="data/seval_qwen3_8b_dataset/train"
TEST_DATA_DIR="data/seval_qwen3_8b_dataset/test"

# 设置保存目录
SAVE_DIR="ckpts/seval_qwen3_8b_sample"

# 设置日志文件
LOG_FILE="log_test.txt"

# 禁用 Python 输出缓冲，确保实时看到输出
export PYTHONUNBUFFERED=1

# 运行训练，输出保存到日志文件同时显示在终端
python train.py \
    --train_dataset_dir "$TRAIN_DATA_DIR" \
    --test_dataset_dir "$TEST_DATA_DIR" \
    --model_name "$MODEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --batch_size 1 \
    --gradient_acc_steps 32 \
    --max_length 1024 \
    --idx_layer 32 \
    --lr 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 10 \
    > "$LOG_FILE" 2>&1
