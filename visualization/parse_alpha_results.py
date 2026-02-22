#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析不同alpha值下的评估结果，生成response level和streaming level的CSV文件
"""

import os
import csv
import re
from pathlib import Path

def _parse_section_metrics(text):
    """
    解析 Response level 或 Streaming level 的表格，提取 7 个指标
    返回: [accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1]
    """
    accuracy = None
    macro_precision, macro_recall, macro_f1 = None, None, None
    weighted_precision, weighted_recall, weighted_f1 = None, None, None

    for line in text.split('\n'):
        line = line.strip()
        # accuracy 行: "accuracy" 后跟数字
        if line.startswith('accuracy') and not line.startswith('weighted'):
            m = re.search(r'accuracy\s+([\d.]+)', line)
            if m:
                accuracy = float(m.group(1))
        # macro avg 行
        elif 'macro avg' in line:
            parts = line.split()
            if len(parts) >= 4:
                macro_precision = float(parts[2])
                macro_recall = float(parts[3])
                macro_f1 = float(parts[4])
        # weighted avg 行
        elif 'weighted avg' in line:
            parts = line.split()
            if len(parts) >= 4:
                weighted_precision = float(parts[2])
                weighted_recall = float(parts[3])
                weighted_f1 = float(parts[4])

    if all(v is not None for v in [accuracy, macro_precision, macro_recall, macro_f1,
                                    weighted_precision, weighted_recall, weighted_f1]):
        return [accuracy, macro_precision, macro_recall, macro_f1,
                weighted_precision, weighted_recall, weighted_f1]
    return None


def parse_eval_log(file_path, alpha):
    """
    解析eval_log.txt文件，从 Response level 和 Streaming level 表格中提取指标
    alpha: 从目录名获取的 alpha 值

    返回: (alpha, response_metrics, streaming_metrics) 或 (None, None, None) 解析失败时
    response_metrics: [accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1]
    streaming_metrics: 同上
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分隔符切分为 Response level 和 Streaming level 两部分
    response_match = re.search(
        r'[-]+Response level[-]+\s*(.*?)(?=[-]+Streaming level[-]+|$)', content, re.DOTALL
    )
    streaming_match = re.search(
        r'[-]+Streaming level[-]+\s*(.*?)(?=[-]+Response level[-]+|$)', content, re.DOTALL
    )

    if not response_match or not streaming_match:
        return None, None, None

    response_metrics = _parse_section_metrics(response_match.group(1))
    streaming_metrics = _parse_section_metrics(streaming_match.group(1))

    if response_metrics and streaming_metrics:
        return alpha, response_metrics, streaming_metrics
    return None, None, None

def main():
    base_dir = Path("/home/ruqi/public/Kelp-ruqi-test/ckpts/seval_qwen3_8b_alpha_exp")
    
    # 定义指标名称
    response_metric_names = [
        'accuracy',
        'precision-macro',
        'recall-macro',
        'f1-macro',
        'precision-weighted-ave',
        'recall-weighted-ave',
        'f1-weighted-ave'
    ]
    
    streaming_metric_names = [
        'accuracy',
        'precision-macro',
        'recall-macro',
        'f1-macro',
        'precision-weighted-ave',
        'recall-weighted-ave',
        'f1-weighted-ave'
    ]
    
    # 收集所有数据
    results = []
    
    # 遍历所有alpha文件夹
    for alpha_dir in sorted(base_dir.glob("alpha_*")):
        if not alpha_dir.is_dir():
            continue
        
        eval_log_path = alpha_dir / "eval_log.txt"
        if not eval_log_path.exists():
            print(f"警告: {eval_log_path} 不存在，跳过")
            continue

        # 从目录名提取 alpha 值 (如 alpha_0.6 -> 0.6)
        alpha_str = alpha_dir.name.replace("alpha_", "")
        try:
            alpha = float(alpha_str)
        except ValueError:
            print(f"警告: 无法解析 alpha 目录名 {alpha_dir.name}，跳过")
            continue

        alpha, response_metrics, streaming_metrics = parse_eval_log(eval_log_path, alpha)
        
        if alpha is not None:
            results.append({
                'alpha': alpha,
                'response': response_metrics,
                'streaming': streaming_metrics
            })
            print(f"成功解析 alpha={alpha}")
        else:
            print(f"警告: 无法解析 {eval_log_path}")
    
    # 按alpha值排序
    results.sort(key=lambda x: x['alpha'])
    
    # 生成Response level CSV
    response_csv_path = base_dir.parent.parent / "results_response_level.csv"
    with open(response_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ['alpha'] + response_metric_names
        writer.writerow(header)
        # 写入数据
        for result in results:
            row = [result['alpha']] + result['response']
            writer.writerow(row)
    
    print(f"\nResponse level CSV已保存到: {response_csv_path}")
    
    # 生成Streaming level CSV
    streaming_csv_path = base_dir.parent.parent / "results_streaming_level.csv"
    with open(streaming_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ['alpha'] + streaming_metric_names
        writer.writerow(header)
        # 写入数据
        for result in results:
            row = [result['alpha']] + result['streaming']
            writer.writerow(row)
    
    print(f"Streaming level CSV已保存到: {streaming_csv_path}")
    
    # 打印统计信息
    print(f"\n共处理 {len(results)} 个alpha值")
    print(f"Alpha值范围: {results[0]['alpha']} - {results[-1]['alpha']}")

if __name__ == "__main__":
    main()
