# -*- coding: utf-8 -*-
"""
A3C_v2_final.py — 最终修复版A3C

修复内容：
1. 正确输出打败预测对手的动作（而非预测本身）
2. 使用确定性argmax策略（而非随机采样）
3. 添加last_policy记录用于Lipschitz分析
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
    return (opp_action - my_action) % 3  # 0 tie,1 opp wins,2 my win


def _score_delta(outcome: int) -> int:
    if outcome == 2: return +1
    if outcome == 1: return -1
    return 0


def _beat(move: int) -> int:
    """返回打败move的动作"""
    return (move + 1) % 3


def _dirichlet_feats(seq: List[int], alpha: float = 1.0) -> np.ndarray:
    c = np.bincount(seq, minlength=3).astype(np.float32) + float(alpha)
    p = c / c.sum()
    return p


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.do = nn.Dropout(dropout)
        self.policy = nn.Linear(hidden, 3)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.do(F.relu(self.fc2(h)))
        logits = self.policy(h)
        v = self.value(h).squeeze(-1)
        return logits, v


class Train:
    name = "A3C_v2un"

    def __init__(
        self,
        ctx_len: int = 20,
        lr: float = 2e-3,
        gamma: float = 0.9,
        entropy_coef: float = 0.01,
        tau_polyak: float = 0.02,
        temp: float = 1.0,
        use_deterministic: bool = True,  # 新参数：是否使用确定性策略
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)
        self.device = _device()

        self.ctx_len = int(max(5, ctx_len))
        self.temp = float(max(0.1, temp))  # 最小温度0.1
        self.gamma = float(gamma)
        self.entropy_coef = float(max(0.0, entropy_coef))
        self.tau_polyak = float(max(0.0, min(1.0, tau_polyak)))
        self.use_deterministic = use_deterministic

        # 输入特征
        self.input_dim = 3 * self.ctx_len + 3 + 3

        self.model = ActorCritic(self.input_dim).to(self.device)
        self.critic_tgt = ActorCritic(self.input_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.model.state_dict())
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.opp_hist: List[int] = []
        self._last = None
        
        # 用于Lipschitz分析
        self.last_policy = None

    def _state(self) -> torch.Tensor:
        # one-hot window
        t = torch.zeros((self.ctx_len, 3), dtype=torch.float32, device=self.device)
        tail = self.opp_hist[-self.ctx_len:]
        for i, a in enumerate(tail, start=self.ctx_len - len(tail)):
            t[i, a] = 1.0
        onehot_flat = t.flatten()
        # dirichlet频率
        feats = _dirichlet_feats(self.opp_hist[-self.ctx_len:] or self.opp_hist, alpha=1.0)
        freq = torch.tensor(feats, dtype=torch.float32, device=self.device)
        # 最近一步
        last = torch.zeros(3, dtype=torch.float32, device=self.device)
        if len(self.opp_hist) > 0:
            last[self.opp_hist[-1]] = 1.0
        s = torch.cat([onehot_flat, freq, last], dim=0).unsqueeze(0)
        return s

    def _polyak_update(self):
        with torch.no_grad():
            for p, p_tgt in zip(self.model.parameters(), self.critic_tgt.parameters()):
                p_tgt.data.mul_(1.0 - self.tau_polyak).add_(self.tau_polyak * p.data)

    def punches(self, round_idx: Optional[int] = None) -> int:
        s = self._state()
        logits, v = self.model(s)
        
        # 获取对手动作的预测分布
        probs = F.softmax(logits / self.temp, dim=1)
        
        # 记录预测分布（用于Lipschitz分析）
        self.last_policy = probs.squeeze().detach().cpu().numpy()
        
        if self.use_deterministic:
            # 策略1：确定性策略（用于测试/评估）
            # 选择最可能的对手动作
            opp_pred = int(torch.argmax(probs, dim=1).item())
            
            # 用于训练更新的记录
            dist = torch.distributions.Categorical(probs=probs)
            logp = dist.log_prob(torch.tensor(opp_pred, device=self.device))
        else:
            # 策略2：随机策略（用于训练时的探索）
            dist = torch.distributions.Categorical(probs=probs)
            opp_pred = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(opp_pred, device=self.device))
        
        # 输出打败预测对手的动作
        my_action = _beat(opp_pred)
        
        # 记录用于训练更新
        self._last = (s, opp_pred, logp, v.detach())  # 注意：存储opp_pred而非my_action
        
        return my_action

    def play(self, my_action: int, opp_action: int) -> None:
        outcome = _outcome(my_action, opp_action)
        self.score += _score_delta(outcome)
        r = 1.0 if outcome == 2 else (-1.0 if outcome == 1 else 0.0)

        # 状态推进
        self.opp_hist.append(opp_action)

        if self._last is None:
            return
            
        s, opp_pred_stored, logp, v = self._last
        
        with torch.no_grad():
            s2 = self._state()
            _, v_tgt_next = self.critic_tgt(s2)

        target = r + self.gamma * v_tgt_next.item()
        adv = target - v

        # 重新计算损失
        logits, v_cur = self.model(s)
        probs = F.softmax(logits / self.temp, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        
        # 使用存储的预测对手动作计算log概率
        logp_cur = dist.log_prob(torch.tensor(opp_pred_stored, device=self.device))

        policy_loss = -logp_cur * adv.detach()
        value_loss = 0.5 * (v_cur - target)**2

        entropy = - (probs * (probs.clamp_min(1e-8)).log()).sum(dim=1).mean()
        loss = policy_loss + value_loss - self.entropy_coef * entropy

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        # Polyak更新
        self._polyak_update()

        self._last = None

    def batch_play(self, my_actions, opp_actions):
        for a_my, a_opp in zip(my_actions, opp_actions):
            self.play(int(a_my), int(a_opp))

    def getscores(self):
        return self.score

    def save(self, idx: Optional[int] = None) -> None:
        import os
        os.makedirs("models", exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "ctx_len": self.ctx_len,
            "temp": self.temp,
            "gamma": self.gamma,
            "tau_polyak": self.tau_polyak,
            "use_deterministic": self.use_deterministic,
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
        
    def load(self) -> None:
        """Load saved model"""
        import os
        path = f"models/{self.idxname}_agent.pth"
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            if "critic_tgt" in state:
                self.critic_tgt.load_state_dict(state["critic_tgt"])
            if "ctx_len" in state:
                self.ctx_len = state["ctx_len"]
            if "temp" in state:
                self.temp = state["temp"]
            if "use_deterministic" in state:
                self.use_deterministic = state["use_deterministic"]