"""
跳过训练，直接评估已训练的模型
使用方法: python eval_skip_train.py
"""
from eval import evaluate_safety_head
from sklearn.metrics import classification_report
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估训练好的 StreamingSafetyHead 模型（跳过训练）")
    parser.add_argument("--ckpt_path", type=str, default="ckpts/seval_qwen3_8b_sample/model_epoch_0.pt", 
                        help="模型检查点路径")
    parser.add_argument("--test_dataset_dir", type=str, default="data/seval_qwen3_8b_dataset/test",
                        help="测试数据集目录")
    parser.add_argument("--model_name", type=str, default="/home/ruqi/public/Qwen3-8B",
                        help="基础模型路径或 Hugging Face ID")
    parser.add_argument("--idx_layer", type=int, default=32,
                        help="用于特征提取的 transformer 层索引")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="数据加载进程数")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="使用 bfloat16 混合精度")
    
    args = parser.parse_args()

    print("=" * 60)
    print("开始评估模型（跳过训练）...")
    print(f"检查点路径: {args.ckpt_path}")
    print(f"测试数据集: {args.test_dataset_dir}")
    print(f"基础模型: {args.model_name}")
    print("=" * 60)
    print()

    predictions, references = evaluate_safety_head(
        ckpt_path=args.ckpt_path,
        test_dataset_dir=args.test_dataset_dir,
        model_name=args.model_name,
        idx_layer=args.idx_layer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bf16=args.bf16,
        return_legacy_format=True  # 使用旧格式以保持向后兼容
    )

    print('\n' + '=' * 60)
    print('评估结果')
    print('=' * 60)
    print(f'检查点路径: {args.ckpt_path}')
    print('\n-------------Response level--------')
    print(classification_report(references, [pred[-2] for pred in predictions], digits=4))

    print('\n-----------Streaming level-----------')
    print(classification_report(references, [max(pred) for pred in predictions], digits=4))
