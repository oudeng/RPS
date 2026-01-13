# -*- coding: utf-8 -*-
"""
M_v2.py — Markov 转移统计（含突变检测 / 加速遗忘）
改动相对 M_v1：
- 保留一阶 Markov 3×3 转移计数（含拉普拉斯平滑）；
- 使用“近期窗口的对数似然”对比（当前模型 vs 近期模型）进行突变检测；
- 一旦检测为“可能突变”，在一段冷却期内启用**高遗忘率**，快速重写过时转移结构。

符号约定：
- 动作编码：0=Rock, 1=Paper, 2=Scissors
- outcome = (opp - mine) % 3
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

class Train:
    name = "M_v2"

    def __init__(
        self,
        smooth: float = 1.0,       # 拉普拉斯平滑
        base_rho: float = 0.01,    # 基础遗忘率
        rho_high: float = 0.2,     # 突变后加速遗忘率
        change_window: int = 60,   # 近期窗口长度（计算 LLR）
        llr_threshold: float = 5.0,# 突变阈值（近期模型相对当前模型的优势）
        cooldown_steps: int = 60,  # 触发后保持高遗忘的步数
        epsilon: float = 0.02,     # 少量探索
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)

        self.smooth = float(max(0.0, smooth))
        self.base_rho = float(max(0.0, min(1.0, base_rho)))
        self.rho_high = float(max(self.base_rho, min(1.0, rho_high)))
        self.change_window = int(max(10, change_window))
        self.llr_threshold = float(llr_threshold)
        self.cooldown_steps = int(max(1, cooldown_steps))
        self.epsilon = float(max(0.0, min(0.2, epsilon)))

        self.prior = np.ones((3,3), dtype=np.float64) * self.smooth
        self.trans = self.prior.copy()   # 计数矩阵（含先验）
        self.last_opp: Optional[int] = None
        self.opp_hist: List[int] = []
        self.cooldown = 0  # 高遗忘冷却计数
        self.t = 0

    # 可选：配合训练器
    def set_batch_size(self, bs: int):
        pass

    def _row_probs(self, row: int) -> np.ndarray:
        v = self.trans[row].astype(np.float64)
        s = v.sum()
        if s <= 0:
            return np.ones(3, dtype=np.float64) / 3.0
        return v / s

    def _apply_forgetting(self, rho: float):
        # T ← prior + (1-ρ)*(T - prior)
        self.trans = self.prior + (1.0 - rho) * (self.trans - self.prior)

    def _recent_llr(self) -> float:
        """近期窗口 LLR = LL(recent-model) - LL(current-model)"""
        if len(self.opp_hist) < self.change_window + 1:
            return 0.0
        seq = self.opp_hist[-(self.change_window + 1):]
        prev = np.array(seq[:-1], dtype=np.int64)
        curr = np.array(seq[1:], dtype=np.int64)
        # 当前模型概率
        P0_rows = np.array([self._row_probs(int(p)) for p in prev])
        p0 = P0_rows[np.arange(len(curr)), curr]
        # 近期模型：用窗口内计数估计（含平滑）
        recent_counts = np.ones((3,3), dtype=np.float64) * self.smooth
        for p, c in zip(prev, curr):
            recent_counts[int(p), int(c)] += 1.0
        recent_rowsum = recent_counts.sum(axis=1, keepdims=True)
        P1 = recent_counts / recent_rowsum
        p1 = P1[prev, curr]
        # LLR
        eps = 1e-12
        ll0 = float(np.sum(np.log(np.clip(p0, eps, 1.0))))
        ll1 = float(np.sum(np.log(np.clip(p1, eps, 1.0))))
        return ll1 - ll0

    def punches(self, round_idx: Optional[int] = None) -> int:
        if self.last_opp is None:
            # 无上一手：用整体边际分布（按列和）
            colsum = self.trans.sum(axis=0).astype(np.float64)
            if colsum.sum() <= 0:
                pred = random.randint(0, 2)
            else:
                pred = int(np.argmax(colsum))
        else:
            probs = self._row_probs(self.last_opp)
            pred = int(np.argmax(probs))
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        return _beat(pred)

    def play(self, my_action: int, opp_action: int) -> None:
        out = _outcome(my_action, opp_action)
        self.score += _score_delta(out)

        # 突变检测：仅在有足够窗口时计算
        llr = self._recent_llr()
        if llr > self.llr_threshold:
            # 近期模型解释力显著强于当前模型 —— 触发高遗忘
            self.cooldown = self.cooldown_steps

        rho = self.rho_high if self.cooldown > 0 else self.base_rho
        self._apply_forgetting(rho)

        # 更新转移计数
        if self.last_opp is not None:
            self.trans[self.last_opp, opp_action] += 1.0
        self.last_opp = int(opp_action)
        self.opp_hist.append(int(opp_action))

        if self.cooldown > 0:
            self.cooldown -= 1
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
            "trans": self.trans,
            "prior": self.prior,
            "smooth": self.smooth,
            "base_rho": self.base_rho,
            "rho_high": self.rho_high,
            "change_window": self.change_window,
            "llr_threshold": self.llr_threshold,
            "cooldown_steps": self.cooldown_steps,
            "epsilon": self.epsilon,
            "t": self.t,
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
