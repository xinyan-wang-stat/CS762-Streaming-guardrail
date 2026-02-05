#!/usr/bin/env python3
"""
将 CSV 文件转换为 HuggingFace datasets 格式
python convert_csv_to_dataset.py --csv_path data/seval_qwen3_8b_sample.csv --output_dir data/seval_qwen3_8b_dataset
"""
import pandas as pd
from datasets import Dataset
import os
import argparse

def convert_csv_to_dataset(csv_path, output_dir, train_ratio=0.8):
    """
    将 CSV 文件转换为 HuggingFace datasets 格式
    
    Args:
        csv_path: CSV 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例（默认 0.8）
    """
    print(f"读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 检查必要的列
    required_cols = ['prompt', 'response', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 文件缺少必需的列: {col}")
    
    # 确保 label 是整数类型
    df['label'] = df['label'].astype(int)
    
    # 如果存在 cut_index 列，确保是整数类型（-1 表示没有切分点）
    if 'cut_index' in df.columns:
        df['cut_index'] = df['cut_index'].astype(int)
        print(f"检测到 cut_index 列，有效值数量: {(df['cut_index'] >= 0).sum()}")
    
    print(f"总样本数: {len(df)}")
    print(f"标签分布:\n{df['label'].value_counts()}")
    
    # 分割训练集和测试集
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train_ratio)
    
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]
    
    print(f"\n训练集: {len(train_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    
    # 转换为 HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 保存数据集
    print(f"\n保存训练集到: {train_dir}")
    train_dataset.save_to_disk(train_dir)
    
    print(f"保存测试集到: {test_dir}")
    test_dataset.save_to_disk(test_dir)
    
    print("\n转换完成！")
    print(f"训练集目录: {train_dir}")
    print(f"测试集目录: {test_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 CSV 转换为 HuggingFace datasets 格式")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例（默认 0.8）")
    
    args = parser.parse_args()
    convert_csv_to_dataset(args.csv_path, args.output_dir, args.train_ratio)
