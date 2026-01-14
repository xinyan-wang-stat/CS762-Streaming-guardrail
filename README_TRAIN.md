# 训练说明

## 已完成的工作

1. ✅ **数据转换**: 已将 CSV 文件转换为 HuggingFace datasets 格式
   - 训练集: `data/seval_qwen3_8b_dataset/train` (80 样本)
   - 测试集: `data/seval_qwen3_8b_dataset/test` (20 样本)

2. ✅ **模型路径**: 已确认模型在 `/home/ruqi/public/Qwen3-8B`

3. ✅ **训练脚本**: 已创建 `run_train.sh`

## 运行训练

### 方法 1: 使用提供的脚本（推荐）

```bash
cd /home/ruqi/public/Kelp
./run_train.sh
```

### 方法 2: 直接运行 Python 命令

```bash
cd /home/ruqi/public/Kelp
python train.py \
    --train_dataset_dir data/seval_qwen3_8b_dataset/train \
    --test_dataset_dir data/seval_qwen3_8b_dataset/test \
    --model_name /home/ruqi/public/Qwen3-8B \
    --save_dir ckpts/seval_qwen3_8b_sample \
    --batch_size 1 \
    --gradient_acc_steps 32 \
    --max_length 1024 \
    --idx_layer 32 \
    --lr 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 1
```

## 注意事项

1. **GPU 内存**: 确保有足够的 GPU 内存（建议至少 16GB）
2. **训练时间**: 100 个样本的训练时间取决于硬件配置
3. **缓存构建**: 首次运行会构建缓存，可能需要一些时间
4. **模型检查点**: 训练完成后，模型会保存在 `ckpts/seval_qwen3_8b_sample/` 目录

## 参数说明

- `--batch_size`: 批次大小（流式处理要求为 1）
- `--gradient_acc_steps`: 梯度累积步数（实际 batch size = batch_size × gradient_acc_steps）
- `--max_length`: 最大序列长度
- `--idx_layer`: 用于特征提取的 transformer 层索引
- `--lr`: 学习率
- `--num_train_epochs`: 训练轮数

## 如果遇到问题

1. **依赖问题**: 确保已安装所有依赖
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA 内存不足**: 可以减少 `--max_length` 或 `--gradient_acc_steps`

3. **模型加载失败**: 检查模型路径是否正确，确保模型文件完整
