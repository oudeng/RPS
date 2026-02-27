# -*- coding: utf-8 -*-
"""
Rename to LSTM_v2
LSTM_v3 = LSTM_v2_enhanced.py - 增强版LSTM，保持在线学习同时发挥LSTM优势

核心策略：
1. ✅ 保持在线学习（每步训练）
2. ✅ 利用LSTM的长序列优势
3. ✅ 自适应学习率
4. ✅ 隐藏状态管理

(Discussion) 关键洞察
成功的关键因素
- 及时性 > 效率：每步立即训练比批量训练更重要
- 时序性 > 随机性：保持数据的时间顺序
- 简单性 > 复杂性：RPS任务不需要过度正则化
- 适应性 > 稳定性：保持学习率以适应变化

LSTM特有优势
- 更长的序列记忆
- 隐藏状态的持续性
- 更好的长期依赖建模
"""

from typing import List, Optional, Tuple
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
    """增强的LSTM网络"""
    def __init__(self, input_size=3, hidden=80, num_layers=2, dropout=0.05):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers
        
        # LSTM层（2层效果可能更好）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 输出层（轻微dropout）
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 3)
        
        # 初始隐藏状态
        self.init_hidden()
    
    def init_hidden(self, batch_size=1):
        """初始化隐藏状态"""
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        self.hidden = (h0, c0)
    
    def forward(self, x, use_hidden=False):
        """
        前向传播
        use_hidden: 是否使用保存的隐藏状态
        """
        if use_hidden and hasattr(self, 'hidden'):
            out, self.hidden = self.lstm(x, self.hidden)
            # Detach防止梯度累积过长
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        else:
            out, _ = self.lstm(x)
        
        h = out[:, -1, :]
        h = self.dropout(h)
        return self.fc(h)


class Train:
    name = "LSTM_v2un"
    
    def __init__(
        self,
        ctx_len: int = 24,  # LSTM可以利用更长的上下文
        lr: float = 1.5e-3,  # 稍高的初始学习率
        lr_decay: float = 0.99995,  # 非常缓慢的衰减
        hidden: int = 80,  # 稍大的隐藏层
        layers: int = 2,  # 2层LSTM
        dropout: float = 0.05,  # 轻微dropout
        temperature: float = 1.0,
        use_hidden_state: bool = True,  # 利用LSTM的状态记忆
        replay_size: int = 20,  # 小规模回放缓冲
        replay_freq: int = 15,  # 每15步回放
        seed: Optional[int] = None,
        **kwargs
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
        self.lr = lr
        self.lr_decay = lr_decay
        self.use_hidden_state = use_hidden_state
        
        # 网络
        self.net = LSTMNet(
            input_size=3,
            hidden=hidden,
            num_layers=layers,
            dropout=dropout
        ).to(self.device)
        
        # 优化器
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
            betas=(0.9, 0.999)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 历史记录
        self.opp_hist: List[int] = []
        self.my_hist: List[int] = []
        
        # 经验回放（保持时序性）
        self.replay_buffer = deque(maxlen=replay_size)
        self.replay_freq = replay_freq
        
        # 性能监控
        self.recent_losses = deque(maxlen=10)
        self.step_count = 0
        self.win_rate = deque(maxlen=100)
        
        print(f"✓ {self.name} enhanced version initialized")
        print(f"  - ctx_len: {self.ctx_len}, hidden: {hidden}, layers: {layers}")
        print(f"  - use_hidden_state: {use_hidden_state}")
    
    def set_batch_size(self, bs: int):
        self._batch_size = max(1, int(bs))
    
    def _one_hot_seq(self, seq: List[int]) -> torch.Tensor:
        """序列one-hot编码"""
        T = len(seq)
        x = torch.zeros((1, T, 3), dtype=torch.float32, device=self.device)
        for i, a in enumerate(seq):
            x[0, i, a] = 1.0
        return x
    
    def _create_input(self) -> torch.Tensor:
        """创建输入（可以结合自己和对手的历史）"""
        if len(self.opp_hist) < 1:
            return None
        
        # 主要使用对手历史
        ctx = self.opp_hist[-self.ctx_len:]
        
        # 可选：交织自己的历史（实验性）
        # combined = []
        # for i in range(min(len(ctx), len(self.my_hist))):
        #     combined.append(ctx[i])
        #     combined.append(self.my_hist[-(i+1)])
        # ctx = combined[-self.ctx_len:]
        
        return self._one_hot_seq(ctx)
    
    def punches(self, round_idx: Optional[int] = None) -> int:
        """预测对手动作"""
        # 冷启动
        if len(self.opp_hist) < 2:
            return random.randint(0, 2)
        
        x = self._create_input()
        if x is None:
            return random.randint(0, 2)
        
        # 推理
        self.net.eval()
        with torch.no_grad():
            # 使用持续的隐藏状态（LSTM的优势）
            logits = self.net(x, use_hidden=self.use_hidden_state)
            
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            # 自适应探索率
            explore_rate = 0.05
            if len(self.win_rate) > 50:
                recent_win_rate = sum(self.win_rate) / len(self.win_rate)
                if recent_win_rate < 0.4:  # 表现不好时增加探索
                    explore_rate = 0.1
            
            # 选择动作
            if random.random() < (1 - explore_rate):
                opp_pred = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = F.softmax(logits * 0.5, dim=-1)  # 温度采样
                opp_pred = torch.multinomial(probs.squeeze(), 1).item()
        
        action = _beat(opp_pred)
        self.my_hist.append(action)
        return action
    
    def _train_step(self, ctx_seq: List[int], target: int) -> float:
        """单步训练"""
        x = self._one_hot_seq(ctx_seq)
        y = torch.tensor([target], dtype=torch.long, device=self.device)
        
        self.net.train()
        logits = self.net(x, use_hidden=False)  # 训练时不用隐藏状态
        loss = self.loss_fn(logits, y)
        
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        return loss.item()
    
    def _replay_train(self):
        """经验回放 - 优先最新样本"""
        if len(self.replay_buffer) < 5:
            return
        
        # 加权采样：最新的样本权重更高
        weights = np.array([0.9 ** (len(self.replay_buffer) - i - 1) 
                           for i in range(len(self.replay_buffer))])
        weights = weights / weights.sum()
        
        # 选择3-5个样本
        n_samples = min(5, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), 
                                  size=n_samples, 
                                  replace=False,
                                  p=weights)
        
        total_loss = 0
        for idx in indices:
            ctx, target = self.replay_buffer[idx]
            loss = self._train_step(ctx, target)
            total_loss += loss
        
        # 记录平均损失
        self.recent_losses.append(total_loss / n_samples)
    
    def _update_lr(self):
        """自适应学习率调整"""
        # 基础衰减
        for g in self.opt.param_groups:
            g['lr'] = max(g['lr'] * self.lr_decay, self.lr * 0.1)
        
        # 基于损失的调整
        if len(self.recent_losses) >= 10:
            avg_loss = np.mean(self.recent_losses)
            if avg_loss > 1.5:  # 损失过高
                for g in self.opt.param_groups:
                    g['lr'] = min(g['lr'] * 1.02, self.lr * 1.5)
    
    def play(self, my_action: int, opp_action: int) -> None:
        """单步更新"""
        # 更新分数
        outcome = _outcome(my_action, opp_action)
        delta = _score_delta(outcome)
        self.score += delta
        
        # 记录胜率
        self.win_rate.append(1 if delta > 0 else 0)
        
        # 立即训练（核心）
        if len(self.opp_hist) >= 1:
            ctx = self.opp_hist[-self.ctx_len:]
            loss = self._train_step(ctx, opp_action)
            self.recent_losses.append(loss)
            
            # 添加到回放缓冲
            self.replay_buffer.append((ctx[:], opp_action))
            
            # 定期回放和调整学习率
            self.step_count += 1
            if self.step_count % self.replay_freq == 0:
                self._replay_train()
                self._update_lr()
            
            # 每100步重置LSTM隐藏状态（防止累积误差）
            if self.step_count % 100 == 0 and self.use_hidden_state:
                self.net.init_hidden()
        
        # 更新历史
        self.opp_hist.append(opp_action)
    
    def batch_play(self, my_actions, opp_actions):
        """批量更新"""
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
            "optimizer": self.opt.state_dict(),
            "ctx_len": self.ctx_len,
            "temperature": self.temperature,
            "step_count": self.step_count
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
            if "temperature" in state:
                self.temperature = state["temperature"]
            if "step_count" in state:
                self.step_count = state["step_count"]