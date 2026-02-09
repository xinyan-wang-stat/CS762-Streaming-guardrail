import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
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
        
        # Use "cuda:0" to avoid full-disk offload (which raises in accelerate).
        # Use low_cpu_mem_usage=False if OOM on load.
        device_map = "cuda:0" if torch.cuda.is_available() else None
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
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
    first_gt_1_list = []   # first position where label==1 per sample
    first_pred_1_list = []  # first position where pred==1 per sample

    autocast_enabled = bf16 and (device.type == "cuda")

    def first_pos(lst, val=1):
        for i, x in enumerate(lst):
            if x == val:
                return i
        return None

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

            logits = safety_head(feat, assistant_start, )

            preds = logits.argmax(dim=-1)  # (1, T_assistant)

            preds_valid = preds.view(-1).tolist()
            labels_valid = labels.view(-1).tolist()

            predictions.append(preds_valid)
            references.append(labels_valid[-1])
            first_gt_1_list.append(first_pos(labels_valid, 1))
            first_pred_1_list.append(first_pos(preds_valid, 1))

    # Compare first label-1 position (only for harmful responses)
    harmful_indices = [i for i in range(len(references)) if references[i] == 1]
    gt_positions = [first_gt_1_list[i] for i in harmful_indices if first_gt_1_list[i] is not None]
    pred_positions = []
    for i in harmful_indices:
        if first_gt_1_list[i] is not None:
            p = first_pred_1_list[i]
            pred_positions.append(len(predictions[i]) if p is None else p)
    if gt_positions:
        diffs = [p - g for g, p in zip(gt_positions, pred_positions)]
        mae = sum(abs(d) for d in diffs) / len(diffs)
        print("\n-----------First harmful token (label 1) position-----------")
        print(f"Only harmful responses (n={len(gt_positions)}): GT first-1 vs Pred first-1")
        print(f"MAE (|pred_pos - gt_pos|): {mae:.2f}")
        print(f"Mean diff (pred - gt, negative=earlier): {sum(diffs)/len(diffs):.2f}")

    return predictions, references


if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, default="ckpts/Qwen-Qwen3-8B/seval.pt")
    p.add_argument("--test_dataset_dir", type=str, default="data/s_eval/qwen3_8b/test/")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--idx_layer", type=int, default=21)
    args = p.parse_args()

    predictions, references = evaluate_safety_head(
        ckpt_path=args.ckpt_path,
        test_dataset_dir=args.test_dataset_dir,
        model_name=args.model_name,
        idx_layer=args.idx_layer,
        max_length=4096,
        batch_size=1,
        num_workers=2,
        bf16=True
    )

    print('ckpt_path: ', args.ckpt_path)
    print('-------------Response level-------- \n', classification_report(references, [pred[-2] for pred in predictions], digits=4))

    print('\n-----------Streaming level-----------\n', classification_report(references, [max(pred) for pred in predictions], digits=4))

