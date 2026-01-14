import os  # 导入操作系统接口模块，用于文件路径操作
import sys  # 导入系统相关参数和函数
import torch  # 导入PyTorch核心库，用于张量操作和深度学习
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口
import glob  # 导入glob模块，用于文件路径模式匹配

from torch.optim import AdamW  # 从PyTorch优化器模块导入AdamW优化器
from torch.utils.data import Dataset, DataLoader  # 从PyTorch数据工具导入数据集和数据加载器
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # 从transformers库导入自动加载因果语言模型、分词器和配置
from sklearn.metrics import f1_score, classification_report  # 从sklearn导入F1分数和分类报告评估指标
from models import StreamingSafetyHead  # 从models模块导入StreamingSafetyHead模型
import math  # 导入数学模块
from transformers import get_cosine_schedule_with_warmup  # 从transformers导入带预热的余弦学习率调度器


from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import random  # 导入random模块，用于随机数生成
import numpy as np  # 导入numpy库，用于数值计算
import argparse  # 导入argparse模块，用于命令行参数解析

from dataset import SafetyDataset  # 从dataset模块导入SafetyDataset数据集类
from eval import evaluate_safety_head  # 从eval模块导入evaluate_safety_head评估函数


def set_seed(seed: int):  # 定义函数：设置随机种子，确保实验可复现
    random.seed(seed)  # 设置Python内置random模块的随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置当前CUDA设备的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有CUDA设备的随机种子
    torch.backends.cudnn.deterministic = True  # 设置cuDNN为确定性模式，确保结果可复现
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN的自动优化，保证确定性
    torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32矩阵乘法，保证数值精度

    print(f"Random seed set globally to {seed}")  # 打印设置的随机种子值

set_seed(42)  # 调用函数设置全局随机种子为42


def count_parameters(model):  # 定义函数：计算模型的可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计所有需要梯度的参数的元素总数
    return trainable_params / 1_000_000  # 返回参数数量（以百万为单位）


def remove_none_fields(d):  # 定义函数：递归移除字典或列表中的None字段
    if isinstance(d, dict):  # 如果输入是字典
        return {k: remove_none_fields(v) for k, v in d.items() if v is not None}  # 递归处理字典的值，只保留非None的项
    elif isinstance(d, list):  # 如果输入是列表
        return [remove_none_fields(i) for i in d]  # 递归处理列表的每个元素
    else:  # 如果是其他类型
        return d  # 直接返回

def compute_temporal_tv_monotone_loss(logits, valid_mask=None, lam_tv=0.01, lam_mono=0.01):  # 定义函数：计算时序总变差和单调性损失
    # logits: (B, T, C)  # 文档注释：logits的形状为(批次大小, 序列长度, 类别数)
    # valid_mask: (B, T) bool  # 文档注释：有效掩码的形状为(批次大小, 序列长度)，布尔类型
    if logits.size(1) < 2:  # 如果序列长度小于2（无法计算相邻差值）
        return torch.zeros([], device=logits.device, dtype=logits.dtype)  # 返回零损失（标量张量）

    p = torch.softmax(logits, dim=-1)[..., 1]  # 对logits应用softmax，然后取第二个类别（索引1，通常是不安全类别）的概率，形状: (B, T)
    diffs = p[:, 1:] - p[:, :-1]  # 计算相邻时刻概率的差值（当前时刻减去前一时刻），形状: (B, T-1)

    if valid_mask is not None:  # 如果提供了有效掩码
        vm = valid_mask[:, 1:] & valid_mask[:, :-1]  # 计算相邻时刻都有效的掩码（两个时刻都必须有效），形状: (B, T-1)
        diffs = diffs[vm]  # 只保留有效位置的差值

    if diffs.numel() == 0:  # 如果差值张量没有元素（所有位置都被掩码）
        return torch.zeros([], device=logits.device, dtype=logits.dtype)  # 返回零损失

    tv = diffs.abs().mean()  # 计算总变差（Total Variation）：差值的绝对值的平均值，鼓励预测平滑
    mono = torch.relu(-diffs).mean()  # 计算单调性损失：负差值的ReLU的平均值，鼓励概率单调递增（即不安全概率随时间增加）

    return lam_tv * tv + lam_mono * mono  # 返回加权后的总损失：总变差损失和单调性损失的加权和



def train(args):  # 定义训练函数，args为命令行参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据CUDA可用性选择设备（GPU或CPU）
    bf16 = True  # 启用bfloat16混合精度训练

    base_model = AutoModelForCausalLM.from_pretrained(  # 从预训练模型加载基础因果语言模型
        args.model_name,  # 模型名称或路径
        torch_dtype="auto",  # 自动选择数据类型
        device_map="auto"  # 自动分配设备（支持多GPU）
    )
    base_model.eval()  # 将基础模型设置为评估模式（禁用dropout和batch normalization的训练模式）
    for p in base_model.parameters():  # 遍历基础模型的所有参数
        p.requires_grad = False  # 禁用所有参数的梯度计算（冻结基础模型，不进行训练）
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)  # 加载与模型对应的分词器，信任远程代码

    train_dataset = SafetyDataset(  # 创建训练数据集
        dataset_dir=args.train_dataset_dir,  # 训练数据集目录路径
        tokenizer=tokenizer,  # 传入分词器
        base_model=base_model,  # 传入基础模型（用于提取hidden states）
        model_name=args.model_name,  # 模型名称
        device=device,  # 设备类型
        idx_layer=args.idx_layer,  # 提取hidden states的层索引
        max_length=args.max_length,  # 最大序列长度
        build_cache_if_missing=True,  # 如果缓存不存在则构建缓存
        overwrite=False,  # 不覆盖已存在的缓存
        max_build_samples=None  # 构建缓存的最大样本数（None表示全部）
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)  # 创建数据加载器，batch_size为批次大小，shuffle为打乱数据，num_workers为数据加载进程数，pin_memory为固定内存加速数据传输

    input_dim = AutoConfig.from_pretrained(args.model_name).hidden_size  # 从模型配置获取隐藏层维度（用于定义安全检测头的输入维度）

    del base_model  # 删除基础模型对象，释放内存
    del tokenizer  # 删除分词器对象，释放内存
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存


    safety_head = StreamingSafetyHead(  # 创建流式安全检测头模型
            input_dim=input_dim,  # 输入维度（基础模型的hidden size）
            proj_dim=1024,  # 投影维度（注意力层输出维度）
            mem_dim=1024,  # 记忆维度（CfcCell的隐藏状态维度）
            num_labels=2,  # 标签数量（2分类：安全/不安全）
            use_dt=True)  # 使用时间步长（连续时间建模）


    safety_head.to(device=device, dtype=torch.bfloat16)  # 将模型移动到指定设备，并使用bfloat16数据类型
    safety_head.requires_grad = True  # 确保模型参数需要梯度（用于训练）

    print("Total trainable parameters: ", count_parameters(safety_head), 'M')  # 打印模型的可训练参数量（以百万为单位）

    optimizer = AdamW(  # 创建AdamW优化器
        safety_head.parameters(),  # 要优化的模型参数
        lr=args.lr,  # 学习率
        weight_decay=args.weight_decay,  # 权重衰减系数（L2正则化）
        betas=(0.9, 0.95),  # Adam优化器的beta参数（动量系数）
        eps=1e-8  # 数值稳定性的小常数
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)  # 创建交叉熵损失函数，ignore_index=-100表示标签为-100的位置不参与损失计算
    max_grad_norm = 1.0  # 梯度裁剪的最大范数阈值

    max_steps = -1  # 最大训练步数（-1表示不使用步数限制）
    lr_scheduler_type = "cosine"  # 学习率调度器类型（余弦退火）
    warmup_ratio = 0.05  # 预热阶段占总训练步数的比例
    warmup_steps = 0  # 预热步数（0表示使用比例计算）

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_acc_steps)  # 计算每个epoch的更新步数（考虑梯度累积）
    if max_steps is None or max_steps < 0:  # 如果未设置最大步数或设置为负数
        total_training_steps = args.num_train_epochs * num_update_steps_per_epoch  # 总训练步数 = epoch数 × 每epoch的更新步数
    else:  # 如果设置了最大步数
        total_training_steps = max_steps  # 使用设定的最大步数

    if warmup_steps and warmup_steps > 0:  # 如果明确设置了预热步数且大于0
        computed_warmup_steps = warmup_steps  # 使用设定的预热步数
    else:  # 否则
        computed_warmup_steps = int(total_training_steps * warmup_ratio)  # 根据比例计算预热步数

    scheduler = get_cosine_schedule_with_warmup(  # 创建带预热的余弦学习率调度器
        optimizer,  # 优化器对象
        num_warmup_steps=computed_warmup_steps,  # 预热步数
        num_training_steps=total_training_steps  # 总训练步数
    )

    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录，如果已存在则不报错


    global_step = 0  # 初始化全局步数计数器
    completed_steps = 0  # 初始化已完成的更新步数计数器
    safety_head.train()  # 将模型设置为训练模式（启用dropout等训练时的行为）

    for epoch in range(args.num_train_epochs):  # 遍历每个训练epoch
        total_loss = 0.0  # 初始化累计损失（用于梯度累积期间的统计）
        total_tokens = 0  # 初始化累计token数（用于计算准确率）
        total_correct = 0  # 初始化累计正确预测数（用于计算准确率）

        optimizer.zero_grad(set_to_none=True)  # 清零梯度（set_to_none=True可以节省内存）

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")):  # 遍历训练数据，显示进度条
            assert batch["labels"].size(0) == 1, "Current implementation assumes batch_size_per_device=1 for streaming."  # 断言：当前实现假设批次大小为1（流式处理的要求）
            labels = batch["labels"].to(device)  # 将标签移动到指定设备，形状: (1, T_assistant)
            feat = batch['embeddings'].to(device)  # 将嵌入向量从CPU移动到指定设备，形状: (seq, hidden)

            assistant_start = batch['assistant_start']  # 获取助手序列的起始位置
            if isinstance(assistant_start, (list, tuple)):  # 如果是列表或元组
                assistant_start = assistant_start[0]  # 取第一个元素
            if isinstance(assistant_start, torch.Tensor):  # 如果是张量
                assistant_start = int(assistant_start.item())  # 转换为整数
            else:  # 其他情况
                assistant_start = int(assistant_start)  # 直接转换为整数

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):  # 启用混合精度训练（自动类型转换）

                # 前向传播，获取每个token的logits，形状: [Bs, N, D]（批次大小，序列长度，类别数）
                logits = safety_head(feat, assistant_start)  

                loss_ce = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))  # 计算交叉熵损失：将logits和labels展平后计算

                anchor_mask = (labels != -100)  # 创建有效标签掩码（标签不为-100的位置），形状: (B, T_assistant)
                reg_mask = torch.ones_like(anchor_mask, dtype=torch.bool)  # 创建正则化掩码（所有位置都有效），形状: (B, T_assistant)
                loss_smooth = compute_temporal_tv_monotone_loss(  # 计算时序平滑损失（总变差和单调性）
                            logits, valid_mask=reg_mask, lam_tv=0.01, lam_mono=0.01  # 传入logits和掩码，总变差权重0.01，单调性权重0.01
                            )

                loss = loss_ce + loss_smooth  # 总损失 = 交叉熵损失 + 平滑损失


                loss = loss / args.gradient_acc_steps  # 将损失除以梯度累积步数（梯度累积的归一化）

            loss.backward()  # 反向传播，计算梯度

            with torch.no_grad():  # 禁用梯度计算（用于统计和评估）
                total_loss += loss.item()  # 累计损失值（用于后续计算平均损失）
                preds = logits.argmax(dim=-1)  # 获取预测类别（取logits最后一个维度的最大值索引），形状: (1, T_assistant)
                mask = (labels != -100)  # 创建有效标签掩码（忽略-100标签）
                correct = (preds[mask] == labels[mask]).sum().item()  # 统计正确预测的token数量
                total_correct += correct  # 累计正确预测数
                total_tokens += mask.sum().item()  # 累计有效token数

            if (step + 1) % args.gradient_acc_steps == 0:  # 如果达到梯度累积步数（需要更新参数）
                torch.nn.utils.clip_grad_norm_(safety_head.parameters(), max_grad_norm)  # 梯度裁剪：将梯度范数限制在max_grad_norm以内，防止梯度爆炸
                optimizer.step()  # 执行优化器步骤（更新模型参数）
                scheduler.step()  # 更新学习率调度器（调整学习率）
                optimizer.zero_grad(set_to_none=True)  # 清零梯度，为下一轮梯度累积做准备

                completed_steps += 1  # 已完成的更新步数加1
                global_step += 1  # 全局步数加1

                current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
                avg_loss = total_loss / args.gradient_acc_steps  # 计算平均损失（梯度累积期间的累计损失除以累积步数）
                avg_acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0  # 计算平均准确率（正确预测数除以总token数）
                print(f"Epoch [{epoch+1}/{args.num_train_epochs}], "  # 打印训练信息：当前epoch
                    f"UpdateStep [{completed_steps}/{total_training_steps}], "  # 当前更新步数和总步数
                    f"LR: {current_lr:.2e}, Loss: {avg_loss:.4f}, Acc(token): {avg_acc:.4f}")  # 学习率、损失和准确率

                total_loss = 0.0  # 重置累计损失
                total_correct = 0  # 重置累计正确数
                total_tokens = 0  # 重置累计token数

                if max_steps is not None and max_steps > 0 and completed_steps >= max_steps:  # 如果达到最大步数
                    break  # 跳出内层循环

        if max_steps is not None and max_steps > 0 and completed_steps >= max_steps:  # 如果达到最大步数
            print("Reached max_steps. Stopping training.")  # 打印停止训练的信息
            break  # 跳出外层循环

        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")  # 构建检查点文件路径（每个epoch保存一次）
        torch.save(safety_head.state_dict(), ckpt_path)  # 保存模型的状态字典（参数）
        print(f"Saved checkpoint: {ckpt_path}")  # 打印保存的检查点路径

    print("Training complete!")  # 打印训练完成信息

    # 在评估前彻底清理 GPU 内存
    print("\n清理 GPU 内存，准备评估...")
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步所有CUDA操作
    import gc  # 导入垃圾回收模块
    gc.collect()  # 强制进行垃圾回收
    print("GPU 内存清理完成")

    predictions, references = evaluate_safety_head(  # 调用评估函数，评估训练好的模型
        ckpt_path=ckpt_path,  # 检查点路径（最后一个epoch的模型）
        test_dataset_dir=args.test_dataset_dir,  # 测试数据集目录路径
        model_name=args.model_name,  # 基础模型名称
        idx_layer=args.idx_layer,  # 提取hidden states的层索引
        max_length=4096,  # 最大序列长度
        batch_size=1,  # 批次大小（流式处理要求为1）
        num_workers=2,  # 数据加载进程数
        bf16=True  # 启用bfloat16混合精度
    )

    print('ckpt_path: ', ckpt_path)  # 打印检查点路径
    print('-------------Response level-------- \n', classification_report(references, [pred[-2] for pred in predictions], digits=4))  # 打印响应级别（response-level）的分类报告：使用每个预测序列的倒数第二个token作为预测结果

    print('\n-----------Streaming level-----------\n', classification_report(references, [max(pred) for pred in predictions], digits=4))  # 打印流式级别（streaming-level）的分类报告：使用每个预测序列的最大值作为预测结果




def main():  # 定义主函数
    parser = argparse.ArgumentParser(description="Train the StreamingSafetyHead with your model.")  # 创建命令行参数解析器，设置描述信息

    # --- Model & Path ---  # 模型和路径相关的参数
    parser.add_argument(  # 添加命令行参数
        "--train_dataset_dir",  # 参数名称：训练数据集目录
        type=str,  # 参数类型：字符串
        required=True,  # 必需参数
        help="Path to the training dataset."  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--test_dataset_dir",  # 参数名称：测试数据集目录
        type=str,  # 参数类型：字符串
        required=True,  # 必需参数
        help="Path to the test dataset."  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--model_name",  # 参数名称：模型名称
        type=str,  # 参数类型：字符串
        default="Qwen/Qwen3-8B",  # 默认值
        help="Path or Hugging Face ID of the base model."  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--save_dir",  # 参数名称：保存目录
        type=str,  # 参数类型：字符串
        required=True,  # 必需参数
        help="Path to save trained model."  # 帮助信息
    )

    # --- Training recipe ---  # 训练配方相关的参数
    parser.add_argument(  # 添加命令行参数
        "--batch_size",  # 参数名称：批次大小
        type=int,  # 参数类型：整数
        default=1,  # 默认值（流式处理要求为1）
    )
    parser.add_argument(  # 添加命令行参数
        "--gradient_acc_steps",  # 参数名称：梯度累积步数
        type=int,  # 参数类型：整数
        default=32,  # 默认值
        help="batch size."  # 帮助信息（实际是梯度累积步数）
    )
    parser.add_argument(  # 添加命令行参数
        "--max_length",  # 参数名称：最大序列长度
        type=int,  # 参数类型：整数
        default=4096,  # 默认值
        help="the max sequence length."  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--idx_layer",  # 参数名称：层索引
        type=int,  # 参数类型：整数
        default=32,  # 默认值
        help="Index of the transformer layers to use for feature extraction"  # 帮助信息：用于特征提取的transformer层索引
    )
    parser.add_argument(  # 添加命令行参数
        "--lr",  # 参数名称：学习率
        type=float,  # 参数类型：浮点数
        default=5e-5,  # 默认值
        help="learning rate"  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--weight_decay",  # 参数名称：权重衰减
        type=float,  # 参数类型：浮点数
        default=0.1,  # 默认值
        help="weight decay"  # 帮助信息
    )
    parser.add_argument(  # 添加命令行参数
        "--num_train_epochs",  # 参数名称：训练epoch数
        type=int,  # 参数类型：整数
        default=1,  # 默认值
        help="nums of training epochs"  # 帮助信息
    )

    args = parser.parse_args()  # 解析命令行参数，获取参数对象

    train(args)  # 调用训练函数，传入解析后的参数


if __name__ == "__main__":  # 当脚本直接运行时执行
    main()  # 调用主函数
