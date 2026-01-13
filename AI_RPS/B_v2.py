# -*- coding: utf-8 -*-
"""
B_v2.py — Dirichlet 频数法（动态遗忘）
改动相对 B_v1：
- 采用 Dirichlet 后验（3 类）对“对手下一手分布”建模；
- 动态遗忘：基于“近期窗口分布 vs 当前后验分布”的一致性（用 Jensen-Shannon 距离）自适应遗忘率；
- 统一接口：Train.punches / play / batch_play / getscores / save。

符号约定：
- 动作编码：0=Rock, 1=Paper, 2=Scissors
- outcome = (opp - mine) % 3   # 0 平, 1 我方负, 2 我方胜
- 我方出拳 = 克制(预测的对手下一手) = (pred + 1) % 3
"""
from typing import List, Optional
import math
import random
import numpy as np
import torch

def _outcome(my_action: int, opp_action: int) -> int:
    return (opp_action - my_action) % 3

def _score_delta(outcome: int) -> int:
    if outcome == 2: return +1
    if outcome == 1: return -1
    return 0

def _beat(move: int) -> int:
    return (move + 1) % 3

def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon 距离（取值 0..1）"""
    p = np.clip(p.astype(np.float64), eps, 1.0); p /= p.sum()
    q = np.clip(q.astype(np.float64), eps, 1.0); q /= q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # 归一化到 [0,1]：最大为 ln 2
    return float(min(1.0, js / math.log(2.0)))

class Train:
    name = "B_v2"

    def __init__(
        self,
        alpha0: float = 1.0,            # Dirichlet 先验强度
        window: int = 50,               # 近期窗口长度
        rho_min: float = 0.005,         # 最小遗忘
        rho_max: float = 0.2,           # 最大遗忘
        delta_low: float = 0.05,        # 一致性好（JS 小）阈值
        delta_high: float = 0.25,       # 一致性差（JS 大）阈值
        epsilon: float = 0.02,          # 少量探索
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)

        self.alpha0 = float(alpha0)
        self.prior = np.ones(3, dtype=np.float64) * self.alpha0
        self.alpha = self.prior.copy()

        self.window = int(max(10, window))
        self.rho_min = float(max(0.0, rho_min))
        self.rho_max = float(min(1.0, max(self.rho_min, rho_max)))
        self.delta_low = float(max(0.0, delta_low))
        self.delta_high = float(max(self.delta_low + 1e-6, delta_high))
        self.epsilon = float(max(0.0, min(0.2, epsilon)))

        self.opp_hist: List[int] = []
        self.t = 0

    # 可选：配合训练器的批量接口（不需要梯度，顺序更新即可）
    def set_batch_size(self, bs: int):  # 与其它方法保持一致接口
        pass

    def _posterior_mean(self) -> np.ndarray:
        a = self.alpha
        return a / a.sum()

    def _map_js_to_forgetting(self, js: float) -> float:
        """将 JS 距离映射到遗忘率 ρ（分段线性）"""
        if js <= self.delta_low:
            return self.rho_min
        if js >= self.delta_high:
            return self.rho_max
        # 线性插值
        r = (js - self.delta_low) / (self.delta_high - self.delta_low)
        return self.rho_min + r * (self.rho_max - self.rho_min)

    def punches(self, round_idx: Optional[int] = None) -> int:
        # 冷启动：少量步随机
        if len(self.opp_hist) < 2:
            return random.randint(0, 2)
        p = self._posterior_mean()
        pred_opp = int(np.argmax(p))
        # epsilon 随机
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        return _beat(pred_opp)

    def _apply_forgetting(self, rho: float):
        # α ← α0 + (1-ρ)*(α-α0)
        self.alpha = self.prior + (1.0 - rho) * (self.alpha - self.prior)

    def play(self, my_action: int, opp_action: int) -> None:
        out = _outcome(my_action, opp_action)
        self.score += _score_delta(out)

        # 动态遗忘率：比较“近期窗口分布”与“当前后验分布”
        if len(self.opp_hist) >= 5:
            tail = self.opp_hist[-self.window:]
            if tail:
                cnt = np.bincount(tail, minlength=3).astype(np.float64)
                p_recent = cnt / cnt.sum()
            else:
                p_recent = np.ones(3, dtype=np.float64) / 3.0
            p_post = self._posterior_mean()
            js = _js_divergence(p_recent, p_post)
            rho = self._map_js_to_forgetting(js)
        else:
            rho = self.rho_min  # 初期保留记忆
        # 先遗忘，再按当前观测加一
        self._apply_forgetting(rho)
        self.alpha[opp_action] += 1.0

        self.opp_hist.append(int(opp_action))
        self.t += 1

    def batch_play(self, my_actions, opp_actions):
        for a_my, a_opp in zip(my_actions, opp_actions):
            self.play(int(a_my), int(a_opp))

    def getscores(self):
        return self.score

    def save(self, idx: Optional[int] = None) -> None:
        import os
        os.makedirs("models", exist_ok=True)
        state = {
            "alpha": self.alpha,
            "prior": self.prior,
            "window": self.window,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "delta_low": self.delta_low,
            "delta_high": self.delta_high,
            "epsilon": self.epsilon,
            "t": self.t,
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
