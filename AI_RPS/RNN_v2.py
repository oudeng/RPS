# -*- coding: utf-8 -*-
"""
RNN_v2: RNN_v2_optimized.py - 优化版RNN，保持在线学习优势

核心改进：
1. ✅ 保持每步训练（在线学习）
2. ✅ 使用最新数据（时序性）  
3. ✅ 适度正则化（防止过拟合）
4. ✅ 固定学习率（持续适应）
5. ✅ 批量训练选项（提高效率）
"""

from typing import List, Optional
import random
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


def _outcome(my_action: int, opp_action: int) -> int:
    return (opp_action - my_action) % 3


def _score_delta(outcome: int) -> int:
    if outcome == 2: return +1
    if outcome == 1: return -1
    return 0


def _beat(move: int) -> int:
    return (move + 1) % 3


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRUNet(nn.Module):
    """优化的GRU网络"""
    def __init__(self, input_size=3, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        
        # GRU层 - 默认单层，无dropout（与v1一致）
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 3)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, hidden=None):
        """
        x: [B, T, 3]
        hidden: optional hidden state
        """
        out, hidden = self.gru(x, hidden)  
        h = out[:, -1, :]  # 取最后时刻
        h = self.dropout(h)
        logits = self.fc(h)  
        return logits, hidden


class Train:
    name = "RNN_v2"
    
    def __init__(
        self,
        ctx_len: int = 16,           # 与v1一致
        lr: float = 1e-3,             # 固定学习率
        lr_min: float = 1e-3,         # 不衰减
        warmup_steps: int = 0,        # 无warmup
        hidden: int = 64,             
        layers: int = 1,              # 单层（与v1一致）
        dropout: float = 0.0,         # 无dropout（与v1一致）
        max_buffer: int = 128,        # 小缓冲区（只用于批量模式）
        batch_size: int = 1,          # 默认单步训练
        update_freq: int = 1,         # 每步都训练！
        seed: Optional[int] = None,
        online_mode: bool = True,     # 在线学习模式
        **kwargs  
    ):
        """
        初始化优化的RNN模型
        
        关键改进：
        - online_mode=True: 每步立即训练（关键！）
        - update_freq=1: 不延迟训练
        - 固定学习率: 持续适应
        - 单层GRU: 简单有效
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)
        self.ctx_len = int(max(4, ctx_len))
        self.device = _device()
        
        # 网络
        self.net = GRUNet(
            input_size=3,
            hidden=hidden,
            num_layers=layers,
            dropout=dropout
        ).to(self.device)
        
        # 优化器 - 使用Adam（与v1一致）
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
            betas=(0.9, 0.999)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 学习率设置（固定）
        self.lr_base = lr
        self.lr_min = lr_min
        self.warmup_steps = int(max(0, warmup_steps))
        self.global_step = 0
        
        # 历史记录
        self.opp_hist: List[int] = []
        
        # 训练缓冲区（用于批量模式）
        self.buffer_x: deque = deque(maxlen=max_buffer)
        self.buffer_y: deque = deque(maxlen=max_buffer)
        
        self.batch_size = int(max(1, batch_size))
        self.update_freq = int(max(1, update_freq))
        self.online_mode = online_mode
        
        # Lipschitz分析
        self.last_policy = None
        
        # 性能监控
        self.rounds_played = 0
        self.train_steps = 0
        
        print(f"✓ {self.name} initialized on {self.device}")
        print(f"  - Mode: {'Online' if online_mode else 'Batch'}")
        print(f"  - ctx_len: {self.ctx_len}, hidden: {hidden}, layers: {layers}")
        print(f"  - lr: {lr:.4f}, update_freq: {update_freq}")
    
    def set_batch_size(self, bs: int):
        """设置批量大小"""
        self.batch_size = max(1, int(bs))
    
    def _one_hot_seq(self, seq: List[int]) -> torch.Tensor:
        """序列one-hot编码"""
        T = len(seq)
        x = torch.zeros((1, T, 3), dtype=torch.float32, device=self.device)
        for i, a in enumerate(seq):
            x[0, i, a] = 1.0
        return x
    
    def punches(self, round_idx: Optional[int] = None) -> int:
        """预测对手动作并返回克制动作"""
        # 冷启动
        if len(self.opp_hist) < 2:
            self.last_policy = np.array([1/3, 1/3, 1/3])
            return random.randint(0, 2)
        
        # 构造输入
        ctx = self.opp_hist[-self.ctx_len:]
        x = self._one_hot_seq(ctx)
        
        # 推理
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(x)
            
            # 记录策略分布
            probs = F.softmax(logits, dim=-1)
            self.last_policy = probs.squeeze().cpu().numpy()
            
            # 预测对手动作
            opp_pred = int(torch.argmax(logits, dim=1).item())
        
        return _beat(opp_pred)
    
    def _train_step_online(self, ctx_seq: List[int], target: int):
        """在线训练步骤（与v1相同）"""
        x = self._one_hot_seq(ctx_seq)
        y = torch.tensor([target], dtype=torch.long, device=self.device)
        
        self.net.train()
        logits, _ = self.net(x)
        loss = self.loss_fn(logits, y)
        
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        self.train_steps += 1
    
    def _train_batch(self):
        """批量训练（备选）"""
        if len(self.buffer_y) < self.batch_size:
            return
        
        # 优先使用最新数据（不随机！）
        batch_size = min(self.batch_size, len(self.buffer_y))
        
        # 从后往前取最新的样本
        X_batch = list(self.buffer_x)[-batch_size:]
        y_batch = list(self.buffer_y)[-batch_size:]
        
        # 构造batch tensor
        B = len(X_batch)
        max_len = max(len(seq) for seq in X_batch)
        
        x = torch.zeros((B, max_len, 3), dtype=torch.float32, device=self.device)
        for i, seq in enumerate(X_batch):
            for t, a in enumerate(seq):
                x[i, t, a] = 1.0
        
        y = torch.tensor(y_batch, dtype=torch.long, device=self.device)
        
        # 训练
        self.net.train()
        logits, _ = self.net(x)
        loss = self.loss_fn(logits, y)
        
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        self.train_steps += 1
    
    def play(self, my_action: int, opp_action: int) -> None:
        """单步更新"""
        # 更新分数
        out = _outcome(my_action, opp_action)
        self.score += _score_delta(out)
        
        # 训练（关键：立即训练！）
        if len(self.opp_hist) >= 1:
            ctx = self.opp_hist[-self.ctx_len:]
            
            if self.online_mode:
                # 在线模式：立即训练（与v1相同）
                self._train_step_online(ctx, opp_action)
            else:
                # 批量模式：添加到缓冲区
                self.buffer_x.append(list(ctx))
                self.buffer_y.append(int(opp_action))
                
                # 定期批量训练
                if self.global_step % self.update_freq == 0:
                    self._train_batch()
        
        # 更新历史
        self.opp_hist.append(opp_action)
        self.rounds_played += 1
        self.global_step += 1
    
    def batch_play(self, my_actions, opp_actions):
        """批量更新"""
        if self.online_mode:
            # 在线模式：逐个处理
            for a_my, a_opp in zip(my_actions, opp_actions):
                self.play(int(a_my), int(a_opp))
        else:
            # 批量模式：收集数据后批量训练
            for a_my, a_opp in zip(my_actions, opp_actions):
                out = _outcome(int(a_my), int(a_opp))
                self.score += _score_delta(out)
                
                if len(self.opp_hist) >= 1:
                    ctx = self.opp_hist[-self.ctx_len:]
                    self.buffer_x.append(list(ctx))
                    self.buffer_y.append(int(a_opp))
                
                self.opp_hist.append(int(a_opp))
                self.rounds_played += 1
            
            # 批处理后训练
            if len(self.buffer_y) >= self.batch_size:
                self._train_batch()
    
    def getscores(self):
        return self.score
    
    def save(self, idx: Optional[int] = None) -> None:
        """保存模型"""
        import os
        os.makedirs("models", exist_ok=True)
        
        state = {
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "ctx_len": self.ctx_len,
            "global_step": self.global_step,
            "train_steps": self.train_steps,
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
        
    def load(self) -> None:
        """加载模型"""
        import os
        path = f"models/{self.idxname}_agent.pth"
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.net.load_state_dict(state["model"])
            if "optimizer" in state:
                self.opt.load_state_dict(state["optimizer"])
            if "ctx_len" in state:
                self.ctx_len = state["ctx_len"]
            if "global_step" in state:
                self.global_step = state["global_step"]
            if "train_steps" in state:
                self.train_steps = state["train_steps"]