import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_recall_fscore_support
import pandas as pd
from models import StreamingSafetyHead
from dataset import SafetyDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_from_disk
import json


def evaluate_safety_head(
    ckpt_path,
    test_dataset_dir,
    model_name="Qwen/Qwen3-8B",
    idx_layer=20,
    max_length=4096,
    batch_size=1,
    num_workers=2,
    bf16=True,
    device=None,
    return_legacy_format=False,  # 如果为True，返回旧格式（列表的列表）以保持向后兼容
    ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 在加载模型前清理 GPU 内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()

    build_cache_if_missing = True
    if build_cache_if_missing:
        # 设置 dtype
        if bf16 and torch.cuda.is_available():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,  # 直接指定 dtype
            device_map="auto",  # 使用 "auto" 让 accelerate 自动分配设备
            cache_dir="./hf_cache"  # 使用本地缓存目录
        )
        # 注意：使用 device_map="auto" 时，不要调用 .to()，因为模型已经被自动分配到设备上了
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="./hf_cache")
    else:
        tokenizer = None
        base_model = None

    test_dataset = SafetyDataset(
        dataset_dir=test_dataset_dir,
        tokenizer=tokenizer,
        base_model=base_model,
        model_name=model_name,
        device="cpu" if not build_cache_if_missing else base_model.device,
        idx_layer=idx_layer,
        max_length=max_length,
        build_cache_if_missing=build_cache_if_missing,
        overwrite=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Load original dataset to get prompts and responses
    original_data = load_from_disk(test_dataset_dir)
    
    sample0 = torch.load(test_dataset.files[0], map_location="cpu")
    input_dim = sample0["embeddings"].shape[-1]
    
    safety_head = StreamingSafetyHead(
        input_dim=input_dim,
        proj_dim=1024,
        mem_dim=1024,
        num_labels=2,
        use_dt=True,
    )


    state = torch.load(ckpt_path, map_location="cpu")
    safety_head.load_state_dict(state, strict=True)
    safety_head.to(device=device, dtype=torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32)
    safety_head.eval()
    
    predictions = []
    references = []
    token_level_labels = []  # 存储 token 级别的 ground truth 标签
    token_level_probs = []  # 存储 token 级别的预测概率
    response_types = []  # 存储 response 类型 (benign/malicious)
    benign_prefixes = []  # 存储 benign prefix (label都是0的部分)
    
    autocast_enabled = bf16 and (device.type == "cuda")
    
    sample_idx = 0  # Track the current sample index
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for batch in tqdm(test_loader):
            assert batch["labels"].size(0) == 1, "Current evaluation assumes batch_size=1 for streaming."
    
            labels = batch["labels"].to(device)            # (1, T_assistant)
            feat = batch["embeddings"].to(device)          # (seq, hidden)
    
            assistant_start = batch["assistant_start"]
            if isinstance(assistant_start, (list, tuple)):
                assistant_start = assistant_start[0]
            if isinstance(assistant_start, torch.Tensor):
                assistant_start = int(assistant_start.item())
            else:
                assistant_start = int(assistant_start)
    
            logits = safety_head(feat, assistant_start)
            
            # 计算概率 (softmax over class dimension)
            probs = F.softmax(logits, dim=-1)  # (1, T_assistant, 2)
            # 获取类别1（harmful）的概率
            probs_class1 = probs[:, :, 1]  # (1, T_assistant)
    
            preds = logits.argmax(dim=-1)  # (1, T_assistant) - token级别预测
            # 使用最后一个token的logits作为response级别预测
            response_pred = logits[:, -1, :].argmax(dim=-1)  # (1,) - response级别预测（使用最后一个token的logits）
    
            preds_valid = preds.view(-1).tolist()
            labels_valid = labels.view(-1).tolist()
            probs_valid = probs_class1.view(-1).tolist()  # 转换为列表
            
            # 确定 response type
            response_type = "malicious" if labels_valid[-1] == 1 else "benign"
            
            # Get prompt and response from original dataset
            prompt = original_data[sample_idx]['prompt']
            response = original_data[sample_idx]['response']
            
            # Calculate benign_prefix (tokens with label=0)
            # Find the first token with label=1
            try:
                first_harmful_idx = labels_valid.index(1)
                # Tokenize the response to get token boundaries
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                # Get benign tokens (before first harmful token)
                if first_harmful_idx > 0 and first_harmful_idx <= len(response_tokens):
                    benign_tokens = response_tokens[:first_harmful_idx]
                    benign_prefix = tokenizer.decode(benign_tokens, skip_special_tokens=True)
                else:
                    benign_prefix = ""
            except ValueError:
                # No harmful token found (all labels are 0)
                benign_prefix = response
            
            sample_idx += 1
    
            if return_legacy_format:
                # 旧格式：返回列表的列表（向后兼容）
                predictions.append(preds_valid)
            else:
                # 新格式：返回字典的列表
                predictions.append({
                    'token_level': preds_valid,  # token级别的预测
                    'response_level': response_pred.item()  # response级别的预测（使用最后一个token的logits）
                })
            references.append(labels_valid[-1])
            token_level_labels.append(labels_valid)  # 存储完整的 token 级别标签
            token_level_probs.append(probs_valid)  # 存储 token 级别的预测概率
            response_types.append(response_type)
            benign_prefixes.append(benign_prefix)
       
    return predictions, references, token_level_labels, token_level_probs, response_types, benign_prefixes, original_data


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_class_0': precision[0] if len(precision) > 0 else 0.0,
        'precision_class_1': precision[1] if len(precision) > 1 else 0.0,
        'recall_class_0': recall[0] if len(recall) > 0 else 0.0,
        'recall_class_1': recall[1] if len(recall) > 1 else 0.0,
        'f1_class_0': f1[0] if len(f1) > 0 else 0.0,
        'f1_class_1': f1[1] if len(f1) > 1 else 0.0,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }


def compute_offset_metrics(predictions, references, token_level_labels):
    """
    计算第一个 harmful token 的位置偏移量指标
    
    Args:
        predictions: 预测列表，每个元素是字典，包含 'token_level' (token级别预测列表)
        references: ground truth 标签列表 (response级别的标签)
        token_level_labels: token 级别的 ground truth 标签列表
    
    Returns:
        字典，包含 MAE 和 mean_diff
    """
    import numpy as np
    
    # 找到第一个 ground truth 为 1 的位置和第一个 prediction 为 1 的位置
    first_gt_1_list = []
    first_pred_1_list = []
    
    for i, pred_dict in enumerate(predictions):
        token_preds = pred_dict['token_level']
        token_labels = token_level_labels[i]
        
        # 找到第一个 ground truth 为 1 的位置
        try:
            first_gt_1 = token_labels.index(1)
            first_gt_1_list.append(first_gt_1)
        except ValueError:
            first_gt_1_list.append(None)
        
        # 找到第一个预测为 1 的位置
        try:
            first_pred_1 = token_preds.index(1)
            first_pred_1_list.append(first_pred_1)
        except ValueError:
            first_pred_1_list.append(None)
    
    # 只考虑 harmful 样本 (references[i] == 1)
    harmful_indices = [i for i in range(len(references)) if references[i] == 1]
    
    if len(harmful_indices) == 0:
        return {
            'mae': 0.0,
            'mean_diff': 0.0,
            'num_harmful_samples': 0
        }
    
    # 收集有效的 gt 和 pred 位置
    gt_positions = []
    pred_positions = []
    
    for i in harmful_indices:
        if first_gt_1_list[i] is not None:
            gt_positions.append(first_gt_1_list[i])
            
            # 如果预测中没有找到 1，使用序列长度作为位置（表示预测到最后都没有检测到 harmful）
            p = first_pred_1_list[i]
            token_len = len(predictions[i]['token_level'])
            pred_positions.append(token_len if p is None else p)
    
    # 如果没有有效的 gt_positions，返回空指标
    if len(gt_positions) == 0 or len(pred_positions) == 0:
        return {
            'mae': 0.0,
            'mean_diff': 0.0,
            'num_harmful_samples': len(harmful_indices)
        }
    
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(pred_positions - gt_positions))
    
    # Mean diff: pred - gt (negative means predicted earlier)
    mean_diff = np.mean(pred_positions - gt_positions)
    
    return {
        'mae': float(mae),
        'mean_diff': float(mean_diff),
        'num_harmful_samples': len(gt_positions)
    }


def save_logits_to_jsonl(ckpt_path, token_level_labels, token_level_probs, response_types, benign_prefixes, original_data):
    """
    保存 token 级别的预测概率到 JSONL 文件
    
    Args:
        ckpt_path: checkpoint 路径
        token_level_labels: token 级别的 ground truth 标签列表
        token_level_probs: token 级别的预测概率列表
        response_types: response 类型列表 (benign/malicious)
        benign_prefixes: benign prefix 列表 (label都是0的部分)
        original_data: 原始数据集，包含 prompt 和 response
    """
    output_path = f"{ckpt_path}_logits.jsonl"
    
    with open(output_path, 'w') as f:
        for idx, (gt, pred, response_type, benign_prefix) in enumerate(zip(token_level_labels, token_level_probs, response_types, benign_prefixes)):
            item = {
                'idx': idx,
                'type': response_type,
                'gt': gt,
                'pred': pred,
                'prompt': original_data[idx]['prompt'],
                'benign_prefix': benign_prefix,
                'response': original_data[idx]['response']
            }
            f.write(json.dumps(item) + '\n')
    
    print(f'\n保存 logits 到: {output_path}')


def evaluate_and_get_metrics(ckpt_path, test_dataset_dir, model_name, idx_layer, max_length=4096, batch_size=1, num_workers=2, bf16=True):
    """评估模型并返回结构化的评估指标"""
    predictions, references, token_level_labels, token_level_probs, response_types, benign_prefixes, original_data = evaluate_safety_head(
        ckpt_path=ckpt_path,
        test_dataset_dir=test_dataset_dir,
        model_name=model_name,
        idx_layer=idx_layer,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        bf16=bf16,
        return_legacy_format=False  # 使用新格式
    )
    
    # Response level 评估
    response_preds = [pred['response_level'] for pred in predictions]
    print('-------------Response level-------- \n', classification_report(references, response_preds, digits=4))
    
    # Streaming level 评估
    streaming_preds = [max(pred['token_level']) for pred in predictions]
    print('\n-----------Streaming level-----------\n', classification_report(references, streaming_preds, digits=4))
    
    # Offset 评估
    offset_metrics = compute_offset_metrics(predictions, references, token_level_labels)
    print(f'\n-----------First harmful token (label 1) position (offset)-----------')
    print(f'Only harmful responses (n={offset_metrics["num_harmful_samples"]}): GT first-1 vs Pred first-1')
    print(f'MAE (|pred_pos - gt_pos|): {offset_metrics["mae"]:.2f}')
    print(f'Mean diff (pred - gt, negative=earlier): {offset_metrics["mean_diff"]:.2f}')
    
    response_metrics = compute_metrics(references, response_preds)
    streaming_metrics = compute_metrics(references, streaming_preds)
    
    # 保存 logits 到 JSONL 文件
    save_logits_to_jsonl(ckpt_path, token_level_labels, token_level_probs, response_types, benign_prefixes, original_data)
    
    return {
        'response_level': response_metrics,
        'streaming_level': streaming_metrics,
        'offset': offset_metrics
    }


if __name__=='__main__':
    model_name = "Qwen/Qwen3-8B"
    ckpt_path = "alpha_0.8/model_epoch_29.pt"
    test_dataset_dir = "data/seval_qwen3_8b_dataset/test/"
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
    
    print('\n========== 评估指标汇总 ==========')
    print('Response Level:', metrics['response_level'])
    print('Streaming Level:', metrics['streaming_level'])
    print('Offset:', metrics['offset'])

