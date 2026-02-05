#!/usr/bin/env python3
"""
直接检查数据集中的 cut_index 值和 label 生成逻辑
"""
import torch
from datasets import load_from_disk
import pandas as pd

def check_data_directly():
    # 加载数据集
    dataset_dir = "data/annotated_output_dataset/train"
    print(f"加载数据集: {dataset_dir}")
    data = load_from_disk(dataset_dir)
    
    # 加载原始 CSV
    csv_path = "data/annotated_output.csv"
    df = pd.read_csv(csv_path)
    
    # 检查缓存文件
    import glob
    import os
    cache_dir = os.path.join(dataset_dir, "safety_cache/-nobackup-ruqi-models-Qwen3-8B/idx32_maxlength1024")
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "sample_*.pt")))
    
    print(f"\n找到 {len(cache_files)} 个缓存文件")
    print(f"数据集大小: {len(data)}")
    print(f"CSV 大小: {len(df)}")
    
    # 检查前几个有害样本
    print("\n检查有害样本（label=1 且 cut_index >= 0）:")
    harmful_count = 0
    for i in range(min(10, len(data))):
        info = data[i]
        csv_row = df.iloc[i] if i < len(df) else None
        
        label = info.get('label', -1)
        cut_index = info.get('cut_index', None)
        
        if label == 1 and cut_index is not None and int(cut_index) >= 0:
            harmful_count += 1
            print(f"\n样本 {i}:")
            print(f"  Label: {label}")
            print(f"  Cut_index (数据集): {cut_index} (类型: {type(cut_index)})")
            if csv_row is not None:
                print(f"  Cut_index (CSV): {csv_row.get('cut_index', 'N/A')}")
            
            # 加载缓存文件
            if i < len(cache_files):
                try:
                    obj = torch.load(cache_files[i], map_location="cpu")
                    labels = obj["labels"]
                    print(f"  标签形状: {labels.shape}")
                    print(f"  标签值: 0={((labels == 0).sum().item())}, 1={((labels == 1).sum().item())}, -100={((labels == -100).sum().item())}")
                    print(f"  标签序列（前30个）: {labels[:30].tolist()}")
                    if len(labels) > 30:
                        print(f"  标签序列（后30个）: {labels[-30:].tolist()}")
                except Exception as e:
                    print(f"  加载失败: {e}")
            
            if harmful_count >= 5:
                break
    
    # 检查所有数据中的 cut_index 分布
    print(f"\n{'='*80}")
    print("数据集中的 cut_index 分布:")
    cut_index_values = []
    for i in range(len(data)):
        info = data[i]
        cut_index = info.get('cut_index', None)
        if cut_index is not None:
            try:
                cut_index_values.append(int(cut_index))
            except:
                cut_index_values.append(cut_index)
    
    if cut_index_values:
        import collections
        counter = collections.Counter(cut_index_values)
        print(f"Cut_index 值统计:")
        for val, count in sorted(counter.items())[:20]:  # 只显示前20个
            print(f"  {val}: {count} 次")
    
    # 检查 CSV 中的 cut_index 分布
    print(f"\nCSV 中的 cut_index 分布:")
    if 'cut_index' in df.columns:
        csv_cut_index = df['cut_index'].value_counts().sort_index()
        print(csv_cut_index.head(20))

if __name__ == "__main__":
    check_data_directly()
