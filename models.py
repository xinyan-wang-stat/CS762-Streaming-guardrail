import torch  # 导入PyTorch核心库，用于张量操作和深度学习
import torch.nn as nn  # 导入PyTorch神经网络模块，用于定义神经网络层
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import re  # 导入正则表达式模块，用于字符串模式匹配
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM  # 从transformers库导入自动加载因果语言模型、分词器和Qwen3模型的类
import torch.nn.functional as F  # 导入PyTorch函数式接口，包含激活函数、损失函数等

def build_custom_attention_mask(seq_len, user_seq_len, device='cpu'):  # 定义函数：构建自定义注意力掩码，用于控制序列中不同部分的可见性
    i = torch.arange(seq_len, device=device).view(-1, 1)  # 创建行索引张量，形状: (L, 1)，表示序列中每个位置的行索引
    j = torch.arange(seq_len, device=device).view(1, -1)  # 创建列索引张量，形状: (1, L)，表示序列中每个位置的列索引
    causal_mask = (i >= user_seq_len) & (i < j)  # 构建因果掩码：助手部分的位置只能看到自己之前的位置（i < j），且位置索引要大于等于用户序列长度
    user_to_assistant_mask = (i < user_seq_len) & (j >= user_seq_len)  # 构建用户到助手的掩码：用户部分的位置可以看到助手部分的位置
    mask = causal_mask | user_to_assistant_mask  # 将两种掩码合并：使用逻辑或操作，得到最终的注意力掩码

    return mask  # 返回构建好的注意力掩码

class AttentionLayer(nn.Module):  # 定义注意力层类，继承自PyTorch的Module基类
    def __init__(self, input_dim=3584, hidden_dim=128):  # 初始化方法，input_dim为输入维度（默认3584），hidden_dim为隐藏层维度（默认128）
        super(AttentionLayer, self).__init__()  # 调用父类的初始化方法
        self.query = nn.Linear(input_dim, hidden_dim)  # 定义查询（Query）线性变换层，将输入维度映射到隐藏维度
        self.key = nn.Linear(input_dim, hidden_dim)  # 定义键（Key）线性变换层，将输入维度映射到隐藏维度
        self.value = nn.Linear(input_dim, hidden_dim)  # 定义值（Value）线性变换层，将输入维度映射到隐藏维度
        self.register_buffer("causal_mask", torch.triu(torch.ones(1, 8192, 8192), diagonal=1).bool())  # 注册一个缓冲区，存储预计算的因果掩码（上三角矩阵，对角线以上为True），最大支持8192长度

    def forward(self, x, user_seq_len=None):  # 定义前向传播方法，x为输入张量，user_seq_len为用户序列长度（可选）
        """
        x: shape (batch_size, seq_len, input_dim)  # 文档字符串：说明输入x的形状为(批次大小, 序列长度, 输入维度)
        """
        batch_size, seq_len, _ = x.size()  # 获取输入张量的批次大小、序列长度和输入维度（使用_忽略最后一个维度）
        Q = self.query(x)  # 通过查询线性层计算查询向量，形状: (batch, seq_len, hidden_dim)
        K = self.key(x)  # 通过键线性层计算键向量，形状: (batch, seq_len, hidden_dim)
        V = self.value(x)  # 通过值线性层计算值向量，形状: (batch, seq_len, hidden_dim)

        # 计算注意力分数：Q和K的转置做矩阵乘法，然后除以K的维度平方根进行缩放，形状: (batch, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / ((K.size(-1) + 1e-6) ** 0.5)  

        if user_seq_len is None:  # 如果用户序列长度未提供
            mask = self.causal_mask[:, :seq_len, :seq_len]  # 使用预计算的因果掩码，截取到当前序列长度，形状: (1, seq_len, seq_len)
        else:  # 如果提供了用户序列长度
            mask = build_custom_attention_mask(seq_len, user_seq_len, x.device)  # 调用自定义函数构建注意力掩码，允许用户部分看到助手部分
        attention_scores = attention_scores.masked_fill(mask, -1e9)  # 将掩码位置填充为极小的负数（-1e9），使得softmax后这些位置的权重接近0

        attention_weights = F.softmax(attention_scores, dim=-1)  # 对注意力分数应用softmax函数，得到注意力权重，在最后一个维度上归一化
        context_vector = torch.matmul(attention_weights, V)  # 使用注意力权重对值向量进行加权求和，得到上下文向量

        return context_vector, attention_weights  # 返回上下文向量和注意力权重


class CfcCell(nn.Module):  # 定义CfcCell类（可能是Continuous-time Feedforward Cell的缩写），继承自PyTorch的Module基类
    def __init__(self, input_dim=128, hidden_dim=128):  # 初始化方法，input_dim为输入维度（默认128），hidden_dim为隐藏状态维度（默认128）
        super(CfcCell, self).__init__()  # 调用父类的初始化方法
        self.input_dim = input_dim  # 保存输入维度
        self.hidden_dim = hidden_dim  # 保存隐藏状态维度

        self.Wz = nn.Parameter(torch.randn(input_dim, hidden_dim))  # 定义更新门（update gate）的输入权重参数，形状: (input_dim, hidden_dim)
        self.Uz = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # 定义更新门的隐藏状态权重参数，形状: (hidden_dim, hidden_dim)
        self.bz = nn.Parameter(torch.zeros(hidden_dim))  # 定义更新门的偏置参数，初始化为零向量，形状: (hidden_dim,)

        self.Wr = nn.Parameter(torch.randn(input_dim, hidden_dim))  # 定义重置门（reset gate）的输入权重参数，形状: (input_dim, hidden_dim)
        self.Ur = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # 定义重置门的隐藏状态权重参数，形状: (hidden_dim, hidden_dim)
        self.br = nn.Parameter(torch.zeros(hidden_dim))  # 定义重置门的偏置参数，初始化为零向量，形状: (hidden_dim,)

        self.Wh = nn.Parameter(torch.randn(input_dim, hidden_dim))  # 定义候选隐藏状态的输入权重参数，形状: (input_dim, hidden_dim)
        self.Uh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # 定义候选隐藏状态的隐藏状态权重参数，形状: (hidden_dim, hidden_dim)
        self.bh = nn.Parameter(torch.zeros(hidden_dim))  # 定义候选隐藏状态的偏置参数，初始化为零向量，形状: (hidden_dim,)

        self.reset_parameters()  # 调用参数重置方法，初始化参数

    def reset_parameters(self):  # 定义参数重置方法，用于初始化模型参数
        for weight in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]:  # 遍历所有权重参数矩阵
            torch.nn.init.orthogonal_(weight)  # 使用正交初始化方法初始化权重矩阵，有助于训练稳定性

    def forward(self, x, h_prev, dt):  # 定义前向传播方法，x为当前输入，h_prev为前一个隐藏状态，dt为时间步长
        z = torch.sigmoid(torch.matmul(x, self.Wz) + torch.matmul(h_prev, self.Uz) + self.bz)  # 计算更新门：对输入和隐藏状态的线性组合应用sigmoid激活函数
        r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_prev, self.Ur) + self.br)  # 计算重置门：对输入和隐藏状态的线性组合应用sigmoid激活函数
        h_hat = torch.tanh(torch.matmul(x, self.Wh) + torch.matmul(r * h_prev, self.Uh) + self.bh)  # 计算候选隐藏状态：使用重置门控制前一个隐藏状态的信息流，然后应用tanh激活函数
        h_new = (1 - z) * h_prev + z * h_hat  # 计算新的隐藏状态：使用更新门在旧隐藏状态和候选隐藏状态之间进行插值

        return h_new + dt * (h_new - h_prev)  # 返回更新后的隐藏状态，加上时间步长乘以变化量（连续时间建模的离散化）


class StreamingSafetyHead(nn.Module):  # 定义流式安全检测头类，继承自PyTorch的Module基类
    def __init__(self,  # 初始化方法
             input_dim=4096,  # 输入维度，默认4096
             proj_dim=512,  # 投影维度，默认512
             mem_dim=512,  # 记忆维度，默认512
             num_labels=1,  # 标签数量，默认1
             use_dt=False,  # 是否使用时间步长，默认False
             dt=1.0,  # 时间步长值，默认1.0
             dropout=0.1,  # Dropout比率，默认0.1（虽然定义了但未使用）
         ):
        super().__init__()  # 调用父类的初始化方法
        self.attention = AttentionLayer(input_dim, proj_dim)  # 创建注意力层，将输入维度投影到投影维度

        d_in = proj_dim  # 设置内部输入维度为投影维度
        self.cfc = CfcCell(d_in, mem_dim)  # 创建CfcCell单元，用于处理序列并维护记忆状态
        self.mem_head = nn.Linear(mem_dim, num_labels)  # 创建线性分类头，将记忆维度映射到标签数量（token scorer）
        self.holistic_head = nn.Linear(mem_dim, num_labels)  # 创建整体分类头，将记忆维度映射到标签数量（response scorer）

        self.prefix_to_h = nn.Sequential(  # 创建前缀到隐藏状态的转换网络，使用Sequential容器
            nn.Linear(d_in, mem_dim),  # 线性层：将投影维度映射到记忆维度
            nn.Tanh()  # Tanh激活函数：将输出限制在[-1, 1]范围内
        )
        self.prefix_scorer = nn.Linear(d_in, 1, bias=False)  # 创建前缀评分器，用于计算用户前缀的注意力权重，无偏置项

        self._h = None  # 初始化隐藏状态为None，形状: (B, mem_dim)

    def reset_state(self, batch_size=None, device=None, dtype=None):  # 定义重置状态方法，用于清空或初始化隐藏状态
        self._h = None  # 将隐藏状态设置为None
        if batch_size is not None:  # 如果提供了批次大小
            if device is None:  # 如果未提供设备
                device = next(self.parameters()).device  # 从模型参数中获取设备
            if dtype is None:  # 如果未提供数据类型
                dtype = next(self.parameters()).dtype  # 从模型参数中获取数据类型
            self._h = torch.zeros(batch_size, self.cfc.hidden_dim, device=device, dtype=dtype)  # 创建零初始化的隐藏状态张量

    def _ensure_state(self, x):  # 定义确保状态方法，用于检查并初始化隐藏状态
        if self._h is None or self._h.size(0) != x.size(0) or self._h.device != x.device or self._h.dtype != x.dtype:  # 如果隐藏状态不存在，或批次大小不匹配，或设备不匹配，或数据类型不匹配
            self.reset_state(batch_size=x.size(0), device=x.device, dtype=x.dtype)  # 重置隐藏状态，使用输入x的批次大小、设备和数据类型

    @torch.no_grad()  # 装饰器：禁用梯度计算，用于推理阶段
    # 定义使用前缀初始化方法，assist_len为助手序列长度，user_hidden为用户隐藏状态，stride为步长（未使用）
    def init_with_prefix(self, assist_len, user_hidden, stride=64):  

        device, dtype = user_hidden.device, user_hidden.dtype  # 获取用户隐藏状态的设备和数据类型
        B, U, D = user_hidden.shape  # 解包用户隐藏状态的形状：批次大小B，用户序列长度U，特征维度D
        self.reset_state(batch_size=B, device=device, dtype=dtype)  # 重置隐藏状态，使用用户隐藏状态的批次大小、设备和数据类型

        times = torch.linspace(0, 1, steps=assist_len).to(device)  # 创建时间序列，从0到1均匀分布，步数为助手序列长度
        self.dt = torch.zeros_like(times)  # 创建与时间序列形状相同的零张量，用于存储时间步长
        self.dt[1:] = times[1:] - times[:-1]  # 计算相邻时间点之间的差值，得到时间步长序列

        x = user_hidden  # 将用户隐藏状态赋值给x

        scores = self.prefix_scorer(x).squeeze(-1)  # 计算用户前缀的评分，然后压缩最后一个维度，形状: (B, U)
        weights = torch.softmax(scores, dim=1)  # 对评分应用softmax，得到注意力权重，在用户序列维度上归一化，形状: (B, U)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # 使用注意力权重对用户隐藏状态进行加权池化，形状: (B, d_in)
        self._h = self.prefix_to_h(pooled)  # 将池化后的特征通过前缀转换网络，得到初始隐藏状态
        return  # 返回（无返回值）


    def step(self, x_t, t):  # 定义单步前向传播方法，x_t为当前时刻的输入特征，t为当前时间步索引

        self._ensure_state(x_t)  # 确保隐藏状态已正确初始化
        self._h = self.cfc(x_t, self._h, self.dt[t])  # 通过CfcCell更新隐藏状态，使用当前输入、前一个隐藏状态和时间步长
        mem_logits = self.mem_head(self._h)  # 通过记忆分类头计算当前时刻的logits，形状: (B, C)

        return mem_logits  # 返回当前时刻的logits

    def forward(self, x, assistant_start, is_multi=False, return_holistic=False):  # 定义前向传播方法，x为输入特征，assistant_start为助手序列的起始位置，is_multi为是否多任务（未使用），return_holistic为是否返回response级别logits

        feat =self.attention(x)[0]  # 通过注意力层处理输入，取第一个返回值（context_vector）作为特征

        seq_total = feat.shape[1]  # 获取特征的总序列长度
        run_len = seq_total - assistant_start  # 计算助手序列的长度（总长度减去起始位置）
        prefix = feat[:, :assistant_start, :]  # 提取用户前缀部分：从开始到助手起始位置的特征
        self.init_with_prefix(run_len, prefix)  # 使用用户前缀初始化隐藏状态，传入助手序列长度和前缀特征

        logits = torch.cat([self.step(feat[:, assistant_start+t, :], t).unsqueeze(1) for t in range(run_len)], dim=1)  # 对助手序列的每个时间步调用step方法，将所有时刻的logits拼接起来，形状: (B, run_len, num_labels)

        if return_holistic:  # 如果需要返回response级别的logits
            # 使用最后一个token的隐藏状态来计算response级别的logits
            holistic_logits = self.holistic_head(self._h)  # 形状: (B, num_labels)
            return logits, holistic_logits  # 返回token级别和response级别的logits
        else:
            return logits  # 返回所有时刻的logits



