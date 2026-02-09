#!/bin/bash
# 训练脚本

# 设置模型路径（本地路径）
#MODEL_PATH="/home/ruqi/public/Qwen3-8B"
MODEL_PATH="/nobackup/ruqi/models/Qwen3-8B"

# 设置数据路径
TRAIN_DATA_DIR="data/annotated_output_dataset/train"
TEST_DATA_DIR="data/annotated_output_dataset/test"

# 设置保存目录
SAVE_DIR="ckpts/seval_qwen3_8b_sample"

# 设置损失权重参数（token级别损失的权重，response级别损失的权重为 1-alpha）
#ALPHA=0 # only response_level CE
#ALPHA=1 # only token_level CE
ALPHA=0.5

# 设置日志文件
LOG_FILE="log_test.txt"

# 禁用 Python 输出缓冲，确保实时看到输出
export PYTHONUNBUFFERED=1

# 检查是否有可用的 unbuffer 工具
# 优先使用 stdbuf（系统自带，更可靠）
if command -v stdbuf > /dev/null 2>&1; then
    # 使用 stdbuf（系统自带，跨平台兼容，禁用行缓冲）
    # -oL: stdout 行缓冲模式（实时输出）
    # -eL: stderr 行缓冲模式（实时输出）
    UNBUFFER_CMD="stdbuf -oL -eL"
elif command -v unbuffer > /dev/null 2>&1; then
    # 使用 unbuffer（如果可用）
    UNBUFFER_CMD="unbuffer"
else
    # 如果都没有，只使用 PYTHONUNBUFFERED（可能不够实时）
    UNBUFFER_CMD=""
    echo "警告: 未找到 stdbuf 或 unbuffer，输出可能不是完全实时的"
    echo "stdbuf 通常是系统自带的，如果缺失可能需要更新系统工具"
fi

# 运行训练，使用 unbuffer 确保实时输出，同时保存到日志文件
$UNBUFFER_CMD python train.py \
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
    --num_train_epochs 20 \
    --alpha "$ALPHA" \
    | tee "$LOG_FILE" 2>&1
