# -*- coding: utf-8 -*-
"""
Tr_v2_fixed.py — 修复版 Transformer（修复 v2 的关键问题）

主要修复：
1. fc_in 层在 __init__ 中正确初始化，确保被优化器跟踪
2. 修复训练逻辑：确保输入-标签对齐
3. 简化：使用固定 ctx_len（可选渐进式增长）
4. 降低正则化强度，适应 RPS 任务特点
5. 保持与 v3.1 的统一接口
"""

from typing import List, Optional
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _outcome(my_action: int, opp_action: int) -> int:
    return (opp_action - my_action) % 3


def _score_delta(outcome: int) -> int:
    if outcome == 2: return +1
    if outcome == 1: return -1
    return 0


def _beat(move: int) -> int:
    return (move + 1) % 3


class TinyTransformer(nn.Module):
    """修复版 Transformer：输入层在 __init__ 中初始化"""
    def __init__(self, d_model=64, nhead=4, num_layers=2, max_len=32, dropout=0.05):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # ✅ 修复：在 __init__ 中定义输入投影层
        self.fc_in = nn.Linear(3, d_model)
        
        # 位置编码
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, 3)
        
        # 初始化
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x_onehot):
        """
        x_onehot: [B, T, 3] one-hot 编码，右对齐
        """
        B, T, C = x_onehot.shape
        assert C == 3, "Input should be one-hot with 3 classes"
        
        # 投影到 d_model
        h = self.fc_in(x_onehot)  # [B, T, d_model]
        
        # 添加位置编码
        h = h + self.pos[:, :T, :]
        
        # Transformer 编码
        h = self.encoder(h)
        
        # 取最后一个 token 预测
        out = self.fc(h[:, -1, :])  # [B, 3]
        
        return out


def _one_hot_seq_right_aligned(seq: List[int], max_len: int, device) -> torch.Tensor:
    """右对齐 one-hot 编码：seq 放在末尾，左侧填充0"""
    x = torch.zeros((1, max_len, 3), dtype=torch.float32, device=device)
    L = min(len(seq), max_len)
    if L > 0:
        for i, a in enumerate(seq[-L:]):
            x[0, max_len - L + i, a] = 1.0
    return x


class Train:
    name = "Tr_v2un"

    def __init__(
        self,
        ctx_len: int = 24,  # ✅ 简化：使用固定长度
        lr: float = 1e-3,
        lr_min: float = 5e-5,
        warmup_steps: int = 200,
        smoothing: float = 0.02,  # ✅ 降低标签平滑
        temperature: float = 1.05,  # ✅ 降低温度
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,  # ✅ 降低 dropout
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)
        self.device = _device()

        # 超参数
        self.ctx_len = int(max(4, ctx_len))
        self.temperature = float(max(0.01, temperature))
        self.smoothing = float(max(0.0, min(0.1, smoothing)))

        # 网络初始化
        self.net = TinyTransformer(
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            max_len=self.ctx_len,
            dropout=dropout
        ).to(self.device)
        
        # 优化器
        self.opt = torch.optim.AdamW(
            self.net.parameters(), 
            lr=lr,
            weight_decay=1e-4  # 轻微 L2 正则
        )
        
        # 学习率调度
        self.lr_base = lr
        self.lr_min = lr_min
        self.warmup_steps = int(max(0, warmup_steps))
        self.global_step = 0

        # 历史记录
        self.opp_hist: List[int] = []
        self.my_hist: List[int] = []

        # 批处理缓冲
        self.buffer_x: List[List[int]] = []
        self.buffer_y: List[int] = []
        self.batch_size = 32
        self.update_freq = 64  # 每收集 N 个样本更新一次

    def set_batch_size(self, bs: int):
        self.batch_size = max(1, int(bs))

    def _update_lr(self):
        """Warmup + 余弦退火学习率"""
        t = self.global_step
        
        if t < self.warmup_steps and self.warmup_steps > 0:
            # Warmup 阶段
            lr = self.lr_base * (t / self.warmup_steps)
        else:
            # 余弦退火
            progress = min(1.0, (t - self.warmup_steps) / float(max(1, 50000)))
            lr = self.lr_min + 0.5 * (self.lr_base - self.lr_min) * (1 + math.cos(math.pi * progress))
        
        for g in self.opt.param_groups:
            g["lr"] = lr

    def punches(self, round_idx: Optional[int] = None) -> int:
        """预测对手动作并返回克制动作"""
        # 冷启动：随机
        if len(self.opp_hist) < 2:
            return random.randint(0, 2)
        
        # 构造输入
        ctx = self.opp_hist[-self.ctx_len:]
        x = _one_hot_seq_right_aligned(ctx, self.ctx_len, self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.net(x) / self.temperature
            opp_pred = int(torch.argmax(logits, dim=-1).item())
        
        return _beat(opp_pred)

    def play(self, my_action: int, opp_action: int) -> None:
        """单步更新"""
        # 更新分数
        outcome = _outcome(my_action, opp_action)
        self.score += _score_delta(outcome)
        
        # ✅ 修复：先保存当前历史作为训练样本
        if len(self.opp_hist) >= 1:
            # 输入：当前时刻之前的历史
            ctx = self.opp_hist[-self.ctx_len:] if len(self.opp_hist) >= self.ctx_len else self.opp_hist[:]
            # 标签：当前对手动作
            self.buffer_x.append(list(ctx))
            self.buffer_y.append(int(opp_action))
        
        # 更新历史（在训练样本采集之后）
        self.opp_hist.append(opp_action)
        self.my_hist.append(my_action)
        
        # 计数器
        self.global_step += 1
        
        # 定期更新
        if self.global_step % self.update_freq == 0:
            self._train_batch()

    def batch_play(self, my_actions, opp_actions):
        """批量更新（保持接口兼容）"""
        for a_my, a_opp in zip(my_actions, opp_actions):
            self.play(int(a_my), int(a_opp))
        
        # 批处理后强制刷新
        self._train_batch(force=True)

    def _train_batch(self, force: bool = False):
        """批量训练"""
        # 检查缓冲区
        if not self.buffer_y:
            return
        
        if len(self.buffer_y) < self.batch_size and not force:
            return
        
        # 取出批量数据
        batch_size = min(self.batch_size, len(self.buffer_y))
        X = self.buffer_x[:batch_size]
        y = self.buffer_y[:batch_size]
        
        # 构造 batch tensor
        B = len(X)
        x_batch = torch.zeros((B, self.ctx_len, 3), dtype=torch.float32, device=self.device)
        
        for i, seq in enumerate(X):
            L = min(len(seq), self.ctx_len)
            for k, action in enumerate(seq[-L:]):
                x_batch[i, self.ctx_len - L + k, action] = 1.0
        
        y_batch = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # 更新学习率
        self._update_lr()
        
        # 前向传播
        logits = self.net(x_batch)
        
        # 损失（标签平滑）
        if self.smoothing > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / 2)
                true_dist.scatter_(1, y_batch.unsqueeze(1), 1.0 - self.smoothing)
            loss = (-true_dist * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, y_batch)
        
        # 反向传播
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        # 清空已处理的样本
        del self.buffer_x[:batch_size]
        del self.buffer_y[:batch_size]

    def getscores(self):
        return self.score

    def save(self, idx: Optional[int] = None) -> None:
        """保存模型"""
        # 先清空缓冲区
        if self.buffer_y:
            self._train_batch(force=True)
        
        import os
        os.makedirs("models", exist_ok=True)
        
        state = {
            "model": self.net.state_dict(),
            "ctx_len": self.ctx_len,
            "temperature": self.temperature,
            "smoothing": self.smoothing,
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")