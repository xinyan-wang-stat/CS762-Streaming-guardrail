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
            device_map="auto"  # 使用 "auto" 让 accelerate 自动分配设备
        )
        # 注意：使用 device_map="auto" 时，不要调用 .to()，因为模型已经被自动分配到设备上了
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
    
    autocast_enabled = bf16 and (device.type == "cuda")
    
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
    
            preds = logits.argmax(dim=-1)  # (1, T_assistant) - token级别预测
            # 使用最后一个token的logits作为response级别预测
            response_pred = logits[:, -1, :].argmax(dim=-1)  # (1,) - response级别预测（使用最后一个token的logits）
    
            preds_valid = preds.view(-1).tolist()
            labels_valid = labels.view(-1).tolist()
    
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
       
    return predictions, references


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


def evaluate_and_get_metrics(ckpt_path, test_dataset_dir, model_name, idx_layer, max_length=4096, batch_size=1, num_workers=2, bf16=True):
    """评估模型并返回结构化的评估指标"""
    predictions, references = evaluate_safety_head(
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
    
    response_metrics = compute_metrics(references, response_preds)
    streaming_metrics = compute_metrics(references, streaming_preds)
    
    return {
        'response_level': response_metrics,
        'streaming_level': streaming_metrics
    }


if __name__=='__main__':
    model_name = "Qwen/Qwen3-8B"
    ckpt_path = "ckpts/Qwen-Qwen3-8B/seval.pt"
    test_dataset_dir = "data/s_eval/qwen3_8b/testset/"
    idx_layer = 21

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

