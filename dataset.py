import os  # 导入操作系统接口模块，用于文件路径操作
import torch  # 导入PyTorch库，用于张量操作和深度学习
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式接口
import glob  # 导入glob模块，用于文件路径模式匹配

from torch.utils.data import Dataset  # 从PyTorch工具包导入Dataset基类
from datasets import load_from_disk  # 从datasets库导入从磁盘加载数据集的函数

from tqdm import tqdm  # 导入tqdm库，用于显示进度条

import random  # 导入random模块，用于随机数生成
import numpy as np  # 导入numpy库，用于数值计算
import tempfile  # 导入tempfile模块，用于创建临时文件

def find_sequence(lst, seq):  # 定义函数：在列表中查找子序列的位置
    n = len(seq)  # 获取子序列的长度
    for i in range(len(lst) - n + 1):  # 遍历列表，查找可能的起始位置
        if lst[i:i+n] == seq:  # 如果找到匹配的子序列
            return i  # 返回起始位置索引
    return -1  # 如果没找到，返回-1


def find_whitespace_token_in_tokenizer_tokens(response_text, cut_index, tokenizer):
    """
    将 whitespace tokenization 的位置映射到 tokenizer token 的位置
    
    Args:
        response_text: 原始的 response 文本
        cut_index: whitespace tokenization 中的位置（0-based），-1 表示没有切分点
        tokenizer: 分词器
    
    Returns:
        cut_token_pos: 在 response 的 tokenizer token 序列中的位置（0-based），-1 表示没有切分点
    """
    if cut_index < 0:  # 没有切分点
        return -1
    
    # 将 response 按空格分割成单词
    words = response_text.split()
    if cut_index >= len(words):
        return -1  # cut_index 超出范围
    
    # 找到 cut_index 指向的单词及其在原始文本中的字符位置
    target_word = words[cut_index]
    
    # 计算 cut_index 之前所有单词的文本（用于找到字符位置）
    if cut_index == 0:
        text_before = ""
    else:
        text_before = ' '.join(words[:cut_index]) + ' '
    
    # 找到目标单词在原始 response 中的字符起始位置
    char_pos = len(text_before)
    
    # 对 response 进行 tokenization，获取每个 token 的字符位置信息
    # 使用 tokenizer 的 encode_plus 获取字符到 token 的映射
    encoding = tokenizer.encode_plus(
        response_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    
    if 'offset_mapping' not in encoding or encoding['offset_mapping'] is None:
        # 如果没有 offset_mapping，使用备用方法
        # 直接 tokenize，然后通过解码找到位置
        token_ids = encoding['input_ids']
        current_char = 0
        for token_idx, token_id in enumerate(token_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            token_len = len(token_text)
            if current_char <= char_pos < current_char + token_len:
                return token_idx
            current_char += token_len
        return len(token_ids) - 1
    else:
        # 使用 offset_mapping 找到字符位置对应的 token
        offset_mapping = encoding['offset_mapping']
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char <= char_pos < end_char:
                return token_idx
            # 如果字符位置在这个 token 之前，返回前一个 token
            if char_pos < start_char:
                return max(0, token_idx - 1)
        
        # 如果没找到，返回最后一个 token 的位置
        return len(offset_mapping) - 1


class SafetyDataset(Dataset):  # 定义SafetyDataset类，继承自PyTorch的Dataset基类
    """
    Per-sample cached dataset:  # 每个样本缓存的数据集
    - cache_dir: {dataset_dir}/safety_cache/{model_name}/idx{idx_layer}maxlength{max_length}/  # 缓存目录路径
    - sample files: sample{orig_idx:08d}.pt, containing:  # 样本文件格式
    {'embeddings': Tensor[seq, hidden], 'assistant_start': int, 'labels': Tensor[T_assistant]}  # 包含嵌入、助手开始位置和标签
    - Build only on rank=0, others wait for barrier and just read.  # 只在rank=0上构建，其他进程等待并只读取
    """
    def __init__(self,  # 初始化方法
        dataset_dir,  # 数据集目录路径
        model_name,  # 模型名称
        tokenizer=None,  # 分词器，可选
        base_model=None,  # 基础模型，可选
        idx_layer: int = 20,  # 使用的层索引，默认为20
        max_length: int = 4096,  # 最大序列长度，默认为4096
        device: str = "cpu",  # 设备类型，默认为CPU
        build_cache_if_missing: bool = False,  # 如果缓存缺失是否构建，默认为False
        overwrite: bool = False,  # 是否覆盖已存在的缓存，默认为False
        max_build_samples: int | None = None,  # 最大构建样本数，None表示全部
        debug_limit: int | None = None  # 调试时限制样本数，None表示全部
        ):
        self.dataset_dir = dataset_dir  # 保存数据集目录路径
        self.model_name = model_name  # 保存模型名称
        self.idx_layer = idx_layer  # 保存层索引
        self.max_length = max_length  # 保存最大序列长度
        self.device = device  # 保存设备类型
        
        # self.user_prompt_marker = [151645, 198, 151644, 77091, 198]  # 用户提示标记（注释掉的代码）
        self.assistant_tokens = '<|im_start|>assistant\n'  # 助手token标记字符串
        self.assistant_end = -1  # 助手回答结束位置，-1表示到序列末尾
        self.num_supervised_token = 10  # 监督学习的token数量
        self.cache_dir = os.path.join(  # 构建缓存目录路径
                dataset_dir,  # 数据集目录
                f"safety_cache/{model_name.replace('/', '-')}/idx{idx_layer}_maxlength{max_length}"  # 缓存子目录路径
            )

        os.makedirs(self.cache_dir, exist_ok=True)  # 创建缓存目录，如果已存在则不报错
        need_build = (len(glob.glob(os.path.join(self.cache_dir, "sample_*.pt"))) == 0)  # 检查是否需要构建缓存（如果缓存目录中没有样本文件）
        if need_build and build_cache_if_missing:  # 如果需要构建且允许构建缓存
            assert tokenizer is not None and base_model is not None, "Building cache requires tokenizer and base_model."  # 断言：构建缓存需要分词器和基础模型
            self._build_cache_per_sample(  # 调用构建缓存方法
                tokenizer=tokenizer,  # 传入分词器
                base_model=base_model,  # 传入基础模型
                overwrite=overwrite,  # 传入是否覆盖参数
                max_build_samples=max_build_samples  # 传入最大构建样本数
            )
    
        self.files = sorted(glob.glob(os.path.join(self.cache_dir, "sample_*.pt")))  # 获取所有样本文件路径并排序
        if debug_limit is not None:  # 如果设置了调试限制
            self.files = self.files[:debug_limit]  # 只取前debug_limit个文件
    
        if len(self.files) == 0:  # 如果没有找到任何缓存文件
            raise FileNotFoundError(f"No cached samples found in {self.cache_dir}. "  # 抛出文件未找到异常
                                    f"Set build_cache_if_missing=True on rank=0 to build first.")  # 提示设置build_cache_if_missing=True来构建缓存
    
    def _build_cache_per_sample(self, tokenizer, base_model, overwrite=False, max_build_samples=None):  # 定义方法：为每个样本构建缓存
        print(f"Building per-sample cache into {self.cache_dir} ...")  # 打印开始构建缓存的信息
        data = load_from_disk(self.dataset_dir)  # 从磁盘加载数据集
        total = len(data) if max_build_samples is None else min(len(data), max_build_samples)  # 计算要处理的样本总数
    
        base_model.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 禁用梯度计算，节省内存和加速
            for i in tqdm(range(total), desc="Build samples"):  # 遍历所有样本，显示进度条
                sample_path = os.path.join(self.cache_dir, f"sample_{i:08d}.pt")  # 构建样本文件路径（8位数字，前面补0）
                if (not overwrite) and os.path.exists(sample_path):  # 如果不覆盖且文件已存在
                    continue  # 跳过该样本
    
                info = data[i]  # 获取第i个样本的数据
                messages = [{'role':'user', 'content': info['prompt']}, {'role':'assistant', 'content': info['response']}]  # 构建消息列表，包含用户提示和助手回答
                text = tokenizer.apply_chat_template(  # 使用分词器将消息列表转换为对话模板格式
                    messages,  # 传入消息列表
                    tokenize=False,  # 不进行分词，返回文本字符串
                    add_generation_prompt=True,  # 添加生成提示符
                    max_length=self.max_length,  # 设置最大长度
                    truncation=True  # 启用截断
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(self.device)  # 将文本转换为模型输入张量，并移动到指定设备
                label = info['label']  # 获取样本标签
    
                output = base_model.generate(  # 调用模型生成方法
                    **model_inputs,  # 解包模型输入参数
                    max_new_tokens=1,  # 最大生成新token数量为1
                    temperature=0,  # 温度参数设为0，使用贪婪解码
                    top_p=1.0,  # top-p采样参数设为1.0
                    top_k=0,  # top-k采样参数设为0
                    do_sample=False,  # 不进行采样，使用贪婪解码
                    repetition_penalty=1.0,  # 重复惩罚系数为1.0（无惩罚）
                    output_hidden_states=True,  # 输出隐藏状态
                    return_dict_in_generate=True  # 以字典形式返回生成结果
                )
                hidden_states = output.hidden_states[0][self.idx_layer]  # 提取第0步（处理输入时）的指定层的hidden states，形状: (1, seq, hidden)
    
                # user_to_assistant_pos = find_sequence(model_inputs.input_ids[0].tolist(), self.user_prompt_marker)  # 查找用户提示标记的位置（注释掉的代码）

                # if user_to_assistant_pos < 0:  # 如果没找到用户提示标记（注释掉的代码）
                #     continue  # 跳过该样本（注释掉的代码）
                # assistant_start = user_to_assistant_pos + len(self.user_prompt_marker)  # 计算助手开始位置（注释掉的代码）
                assistant_ids = tokenizer.encode(self.assistant_tokens)  # 将助手token字符串编码为token ID列表
                assistant_start = find_sequence(model_inputs.input_ids[0].tolist(), assistant_ids) + len(assistant_ids)  # 查找助手token序列的位置，并计算助手回答的起始位置
    
                seq_len = model_inputs.input_ids[:, assistant_start:self.assistant_end].shape[-1]  # 计算助手回答部分的序列长度
                if seq_len <= 0:  # 如果序列长度小于等于0
                    continue  # 跳过该样本
                
                # 获取 cut_index（如果存在）
                cut_index = info.get('cut_index', None)  # 如果没有 cut_index 列，返回 None
                if cut_index is None:
                    # 没有 cut_index 列，使用原来的标签生成逻辑
                    labels = torch.full((1, seq_len), -100, dtype=torch.long, device=self.device)  # 创建标签张量，初始值全为-100（忽略索引）
                    labels[:, :self.num_supervised_token] = 0  # 将前num_supervised_token个token的标签设为0（安全）
                    labels[:, -self.num_supervised_token:] = torch.tensor([label], device=self.device).unsqueeze(1).expand(-1, self.num_supervised_token)  # 将最后num_supervised_token个token的标签设为实际标签值
                else:
                    cut_index = int(cut_index)
                    if cut_index < 0:
                        # cut_index = -1 表示标签为0（安全），所有 token 都是 0
                        labels = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
                    else:
                        # cut_index >= 0，使用新的标签生成逻辑：基于 cut_index
                        # 将 whitespace token 位置映射到 tokenizer token 位置
                        response_text = info['response']
                        cut_token_pos = find_whitespace_token_in_tokenizer_tokens(
                            response_text, cut_index, tokenizer
                        )
                        
                        if cut_token_pos >= 0 and cut_token_pos < seq_len:
                            # cut_index 之前（不包括）的 token = 0
                            # cut_index 及之后（包括）的 token = 1
                            labels = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)  # 初始化为0
                            labels[:, cut_token_pos:] = 1  # cut_token_pos 及之后设为1
                        else:
                            # 如果映射失败，使用原来的逻辑
                            labels = torch.full((1, seq_len), -100, dtype=torch.long, device=self.device)
                            labels[:, :self.num_supervised_token] = 0
                            labels[:, -self.num_supervised_token:] = torch.tensor([label], device=self.device).unsqueeze(1).expand(-1, self.num_supervised_token)
    
                embedding_cpu = hidden_states[0, :self.assistant_end, :].detach().cpu().contiguous()  # 提取hidden states并转移到CPU，形状: (seq, hidden)
                labels_cpu = labels[0].detach().cpu().contiguous()  # 提取标签并转移到CPU，形状: (T_assistant,)
    
                payload = {  # 构建要保存的数据字典
                    "embeddings": embedding_cpu,          # 嵌入向量，形状: (seq, hidden)
                    "assistant_start": int(assistant_start),  # 助手开始位置（转换为整数）
                    "labels": labels_cpu                   # 标签，形状: (T_assistant,)
                }
    
                tmp_fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir)  # 在缓存目录中创建临时文件
                os.close(tmp_fd)  # 关闭临时文件描述符
                torch.save(payload, tmp_path)  # 将数据保存到临时文件
                os.replace(tmp_path, sample_path)  # 原子性地将临时文件重命名为目标文件
    
        print(f"Cache build finished at {self.cache_dir}")  # 打印缓存构建完成的信息
    
    def __len__(self):  # 定义方法：返回数据集大小
        return len(self.files)  # 返回文件列表的长度
    
    def __getitem__(self, idx):  # 定义方法：获取指定索引的样本
        obj = torch.load(self.files[idx], map_location="cpu")  # 从文件加载数据到CPU
        embeddings = obj["embeddings"]            # 提取嵌入向量，形状: (seq, hidden)，CPU张量
        assistant_start = obj['assistant_start']  # 提取助手开始位置
        labels = torch.as_tensor(obj["labels"], dtype=torch.long)  # 将标签转换为长整型张量，形状: (T_assistant,)
        return {  # 返回样本字典
            "embeddings": embeddings,  # 嵌入向量
            "assistant_start": assistant_start,  # 助手开始位置
            "labels": labels  # 标签
        }


