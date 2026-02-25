# -*- coding: utf-8 -*-
"""
A3C.py - GPU优化版Actor-Critic模型（支持批量处理）
"""

import math
import random
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _outcome(my_action: int, opp_action: int) -> int:
    """0=tie, 1=opp wins, 2=my win"""
    return (opp_action - my_action) % 3


def _score_delta(outcome: int) -> int:
    if outcome == 2:
        return +1
    elif outcome == 1:
        return -1
    return 0


def _beat_move(move: int) -> int:
    return (move + 1) % 3


def _argmax_random_tie(x: torch.Tensor) -> int:
    """x: 1D tensor"""
    max_val = torch.max(x)
    idxs = torch.nonzero(x == max_val, as_tuple=False).flatten().tolist()
    return random.choice(idxs) if idxs else int(torch.argmax(x).item())


class ActorCritic(nn.Module):
    def __init__(self, input_dim=30, hidden=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.do = nn.Dropout(dropout)
        self.policy = nn.Linear(hidden, 3)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do(F.relu(self.fc2(x)))
        logits = self.policy(x)
        val = self.value(x).squeeze(-1)
        return logits, val


class Train:
    name = "A3C_v1"

    def __init__(self, ctx_len: int = 10, lr: float = 2e-3, gamma: float = 0.9,
                 entropy_coef: float = 0.01, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.idxname = self.name
        self.agent_scores = torch.tensor([0], dtype=torch.int64)
        self.device = _device()
        self.ctx_len = int(max(5, ctx_len))
        self.model = ActorCritic(input_dim=self.ctx_len * 3).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = float(np.clip(gamma, 0.0, 0.999))
        self.entropy_coef = float(np.clip(entropy_coef, 0.0, 0.05))
        self.opp_hist: List[int] = []
        self._last = None
        
        # 批量处理相关
        self.batch_size = 32
        self.update_buffer: List[Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, int, int]] = []
        
    def set_batch_size(self, batch_size: int):
        """设置批量大小"""
        self.batch_size = max(1, batch_size)

    def _state(self) -> torch.Tensor:
        """Last ctx_len opponent actions one-hot concatenated -> [3*ctx_len]"""
        seq = self.opp_hist[-self.ctx_len:]
        t = torch.zeros((self.ctx_len, 3), dtype=torch.float32, device=self.device)
        for i, a in enumerate(seq, start=self.ctx_len - len(seq)):
            t[i, a] = 1.0
        return t.flatten().unsqueeze(0)  # [1, 3*ctx_len]

    def punches(self, round_idx: Optional[int] = None) -> int:
        s = self._state()
        logits, val = self.model(s)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        action = int(dist.sample().item())
        self._last = (s, action, dist.log_prob(torch.tensor(action, device=self.device)), val.squeeze(0))
        return action

    def play(self, my_action: int, opp_action: int) -> None:
        """单个更新（兼容接口）"""
        out = _outcome(my_action, opp_action)
        self.agent_scores += _score_delta(out)
        
        # 计算奖励
        r = 1.0 if out == 2 else (-1.0 if out == 1 else 0.0)
        
        # 添加到历史
        self.opp_hist.append(opp_action)
        
        if self._last is not None:
            s, a, logp, v = self._last
            # 缓存数据用于批量更新
            self.update_buffer.append((s, a, logp, v, r, opp_action))
            
            # 如果缓存满了，执行批量更新
            if len(self.update_buffer) >= self.batch_size:
                self._batch_update()
            
            self._last = None

    def batch_play(self, batch: List[Tuple[int, int]]) -> None:
        """批量更新（优化接口）"""
        # 收集所有数据
        batch_data = []
        
        for my_action, opp_action in batch:
            out = _outcome(my_action, opp_action)
            self.agent_scores += _score_delta(out)
            r = 1.0 if out == 2 else (-1.0 if out == 1 else 0.0)
            self.opp_hist.append(opp_action)
            
            if self._last is not None:
                s, a, logp, v = self._last
                batch_data.append((s, a, logp, v, r, opp_action))
                self._last = None
        
        # 批量训练
        if batch_data:
            self._batch_train(batch_data)

    def _batch_update(self):
        """执行批量更新"""
        if not self.update_buffer:
            return
        
        self._batch_train(self.update_buffer)
        self.update_buffer.clear()

    def _batch_train(self, batch_data: List[Tuple]):
        """批量训练网络"""
        if not batch_data:
            return
        
        # 准备批量数据
        states = []
        actions = []
        old_log_probs = []
        old_values = []
        rewards = []
        next_states = []
        
        for i, (s, a, logp, v, r, opp_action) in enumerate(batch_data):
            states.append(s.squeeze(0))
            actions.append(a)
            old_log_probs.append(logp)
            old_values.append(v)
            rewards.append(r)
            
            # 计算下一状态（用于价值估计）
            if i < len(batch_data) - 1:
                # 使用批次中的下一个状态
                next_s = batch_data[i + 1][0].squeeze(0)
            else:
                # 使用当前状态计算下一状态
                next_s = self._state().squeeze(0)
            next_states.append(next_s)
        
        # 转换为张量
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        old_values = torch.stack(old_values)
        old_log_probs = torch.stack(old_log_probs)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        
        # 批量前向传播
        logits, values = self.model(states)
        with torch.no_grad():
            _, next_values = self.model(next_states)
        
        # 计算优势和目标价值
        targets = rewards + self.gamma * next_values
        advantages = targets - values
        
        # 策略损失（使用log概率）
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        
        # 价值损失
        value_loss = 0.5 * advantages.pow(2).mean()
        
        # 熵奖励
        entropy = dist.entropy().mean()
        
        # 总损失
        loss = policy_loss + value_loss - self.entropy_coef * entropy
        
        # 批量反向传播
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

    def getscores(self):
        return self.agent_scores

    def save(self, idx: Optional[int] = None) -> None:
        # 确保所有缓存都已处理
        self._batch_update()
        
        state = {"model": self.model.state_dict(), "ctx_len": self.ctx_len}
        torch.save(state, f"models/{self.idxname}_agent.pth")
        print(f"{self.idxname} scores: {int(self.agent_scores.item())}")
