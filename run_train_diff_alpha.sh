#!/bin/bash
# 训练脚本 - 测试不同的 alpha 值

# 设置模型路径（本地路径）
#MODEL_PATH="/home/ruqi/public/Qwen3-8B"
MODEL_PATH="/nobackup/ruqi/models/Qwen3-8B"

# 设置数据路径
TRAIN_DATA_DIR="data/annotated_output_dataset/train"
TEST_DATA_DIR="data/annotated_output_dataset/test"

# 设置基础保存目录
BASE_SAVE_DIR="ckpts/seval_qwen3_8b_alpha_exp"

# 设置要测试的 alpha 值
#ALPHAS=(0 0.2 0.4 0.6 0.8 1.0)
ALPHAS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# 设置日志文件
LOG_FILE="log_alpha_experiment.txt"
RESULTS_FILE="results_alpha_experiment.csv"

# 禁用 Python 输出缓冲，确保实时看到输出
export PYTHONUNBUFFERED=1

# 检查是否有可用的 unbuffer 工具
if command -v stdbuf > /dev/null 2>&1; then
    UNBUFFER_CMD="stdbuf -oL -eL"
elif command -v unbuffer > /dev/null 2>&1; then
    UNBUFFER_CMD="unbuffer"
else
    UNBUFFER_CMD=""
    echo "警告: 未找到 stdbuf 或 unbuffer，输出可能不是完全实时的"
fi

# 初始化结果文件（CSV格式）
echo "alpha,response_accuracy,response_precision_macro,response_recall_macro,response_f1_macro,response_precision_weighted,response_recall_weighted,response_f1_weighted,streaming_accuracy,streaming_precision_macro,streaming_recall_macro,streaming_f1_macro,streaming_precision_weighted,streaming_recall_weighted,streaming_f1_weighted" > "$RESULTS_FILE"

# 循环测试不同的 alpha 值
for ALPHA in "${ALPHAS[@]}"; do
    echo "=========================================="
    echo "开始训练 alpha = $ALPHA"
    echo "=========================================="
    
    # 为每个 alpha 创建独立的保存目录
    SAVE_DIR="${BASE_SAVE_DIR}/alpha_${ALPHA}"
    mkdir -p "$SAVE_DIR"
    
    # 为每个 alpha 创建独立的日志文件
    ALPHA_LOG_FILE="${SAVE_DIR}/train_log.txt"
    
    # 运行训练
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
        | tee "$ALPHA_LOG_FILE" 2>&1
    
    # 检查训练是否成功（检查最后一个epoch的checkpoint是否存在）
    LAST_EPOCH=19  # 第20个epoch（从0开始计数）
    CKPT_PATH="${SAVE_DIR}/model_epoch_${LAST_EPOCH}.pt"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "警告: alpha=$ALPHA 的训练未找到checkpoint文件: $CKPT_PATH"
        echo "跳过评估..."
        continue
    fi
    
    echo "=========================================="
    echo "开始评估 alpha = $ALPHA"
    echo "=========================================="
    
    # 运行评估并提取指标
    EVAL_LOG_FILE="${SAVE_DIR}/eval_log.txt"
    python -c "
import sys
sys.path.insert(0, '.')
from eval import evaluate_and_get_metrics
import json

ckpt_path = '$CKPT_PATH'
test_dataset_dir = '$TEST_DATA_DIR'
model_name = '$MODEL_PATH'
idx_layer = 32

metrics = evaluate_and_get_metrics(
    ckpt_path=ckpt_path,
    test_dataset_dir=test_dataset_dir,
    model_name=model_name,
    idx_layer=idx_layer,
    max_length=4096,
    batch_size=1,
    num_workers=2,
    bf16=True
)

# 保存详细结果到JSON文件
results_json = {
    'alpha': $ALPHA,
    'response_level': metrics['response_level'],
    'streaming_level': metrics['streaming_level']
}
with open('${SAVE_DIR}/metrics.json', 'w') as f:
    json.dump(results_json, f, indent=2)

# 输出CSV格式的结果（追加到结果文件）
r = metrics['response_level']
s = metrics['streaming_level']
print(f\"$ALPHA,{r['accuracy']:.4f},{r['precision_macro']:.4f},{r['recall_macro']:.4f},{r['f1_macro']:.4f},{r['precision_weighted']:.4f},{r['recall_weighted']:.4f},{r['f1_weighted']:.4f},{s['accuracy']:.4f},{s['precision_macro']:.4f},{s['recall_macro']:.4f},{s['f1_macro']:.4f},{s['precision_weighted']:.4f},{s['recall_weighted']:.4f},{s['f1_weighted']:.4f}\")
" | tee "$EVAL_LOG_FILE" >> "$RESULTS_FILE" 2>&1
    
    echo "alpha=$ALPHA 的训练和评估完成"
    echo ""
done

echo "=========================================="
echo "所有实验完成！"
echo "结果已保存到: $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
