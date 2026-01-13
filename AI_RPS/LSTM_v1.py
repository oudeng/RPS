# -*- coding: utf-8 -*-
"""
Renamed to LATM_v1
LSTM_v2_fixed.py - 修复版LSTM，应用RNN_v2的成功经验

关键修复：
1. ✅ 每步立即训练（不延迟）
2. ✅ 使用最新数据（不随机采样）
3. ✅ 简化正则化
4. ✅ 固定或缓慢的学习率变化
5. ✅ 保持LSTM的序列建模优势
"""

from typing import List, Optional
import math
import random
import numpy as np
from collections import deque

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


class LSTMNet(nn.Module):
    """简化的LSTM网络"""
    def __init__(self, input_size=3, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        
        # LSTM层 - 默认单层，无dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 输出层（无dropout）
        self.fc = nn.Linear(hidden, 3)
        # 不做自定义初始化，使用PyTorch默认
    
    def forward(self, x):
        """简化的forward，只返回logits"""
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # 取最后时刻
        return self.fc(h)  # 只返回logits


class Train:
    name = "LSTM_v1"
    
    def __init__(
        self,
        ctx_len: int = 20,  # LSTM可以处理更长的序列
        lr: float = 1e-3,   # 固定学习率
        hidden: int = 64,
        layers: int = 1,    # 简化为单层
        dropout: float = 0.0,  # 无dropout
        temperature: float = 1.0,  # 不做温度缩放
        seed: Optional[int] = None,
        **kwargs  # 忽略额外参数
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)
        self.device = _device()
        
        # 超参数
        self.ctx_len = int(max(4, ctx_len))
        self.temperature = temperature
        
        # 网络（简化版）
        self.net = LSTMNet(
            input_size=3,
            hidden=hidden,
            num_layers=layers,
            dropout=dropout
        ).to(self.device)
        
        # 优化器（使用Adam，不是AdamW）
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr=lr
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 历史记录
        self.opp_hist: List[int] = []
        self._batch_size = 1
        
        # 小的经验缓冲（可选）
        self.recent_buffer = deque(maxlen=10)
        self.use_replay = False  # 默认不使用
        
        print(f"✓ {self.name} fixed version initialized")
        print(f"  - ctx_len: {self.ctx_len}, hidden: {hidden}, layers: {layers}")
        print(f"  - lr: {lr:.4f} (fixed)")
    
    def set_batch_size(self, bs: int):
        self._batch_size = max(1, int(bs))
    
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
            return random.randint(0, 2)
        
        # 构造输入（使用最近的历史）
        ctx = self.opp_hist[-self.ctx_len:]
        x = self._one_hot_seq(ctx)
        
        # 推理
        self.net.eval()
        with torch.no_grad():
            logits = self.net(x)
            
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            # 贪婪选择（大部分时间）
            if random.random() < 0.95:  # 95%贪婪
                opp_pred = int(torch.argmax(logits, dim=-1).item())
            else:
                # 5%探索
                probs = F.softmax(logits, dim=-1)
                opp_pred = torch.multinomial(probs.squeeze(), 1).item()
        
        return _beat(opp_pred)
    
    def _train_step(self, ctx_seq: List[int], target: int):
        """单步训练（核心）"""
        x = self._one_hot_seq(ctx_seq)
        y = torch.tensor([target], dtype=torch.long, device=self.device)
        
        self.net.train()
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
    
    def _replay_train(self):
        """可选的小规模经验回放"""
        if len(self.recent_buffer) < 3:
            return
        
        # 只训练最近的几个样本
        for ctx, target in list(self.recent_buffer)[-3:]:
            self._train_step(ctx, target)
    
    def play(self, my_action: int, opp_action: int) -> None:
        """单步更新 - 关键：每步都立即训练！"""
        # 更新分数
        outcome = _outcome(my_action, opp_action)
        self.score += _score_delta(outcome)
        
        # 立即训练（核心改进）
        if len(self.opp_hist) >= 1:
            ctx = self.opp_hist[-self.ctx_len:]
            self._train_step(ctx, opp_action)
            
            # 可选：添加到小缓冲区
            if self.use_replay:
                self.recent_buffer.append((ctx[:], opp_action))
                # 每10步做一次小规模回放
                if len(self.opp_hist) % 10 == 0:
                    self._replay_train()
        
        # 更新历史
        self.opp_hist.append(opp_action)
    
    def batch_play(self, my_actions, opp_actions):
        """批量更新 - 但仍然逐个处理保持时序性"""
        for a_my, a_opp in zip(my_actions, opp_actions):
            self.play(int(a_my), int(a_opp))
    
    def getscores(self):
        return self.score
    
    def save(self, idx: Optional[int] = None) -> None:
        """保存模型"""
        import os
        os.makedirs("models", exist_ok=True)
        
        state = {
            "model": self.net.state_dict(),
            "ctx_len": self.ctx_len,
            "temperature": self.temperature
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
    
    def load(self) -> None:
        """加载模型"""
        import os
        path = f"models/{self.idxname}_agent.pth"
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.net.load_state_dict(state["model"])
            if "ctx_len" in state:
                self.ctx_len = state["ctx_len"]
            if "temperature" in state:
                self.temperature = state["temperature"]