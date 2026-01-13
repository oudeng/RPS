# -*- coding: utf-8 -*-
"""
B_v1.py - 贝叶斯/狄利克雷后验策略（CPU优化版）
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


class Train:
    name = "B_v1"

    def __init__(self, alpha: float = 1.0, forget: float = 0.98, eps: float = 0.05,
                 seed: Optional[int] = None):
        """
        Dirichlet posterior over opponent move frequencies with exponential forgetting.
        - alpha: prior strength per move
        - forget: forgetting factor in (0,1]; applied to counts each step
        - eps: epsilon-greedy on the final chosen 'my action' to avoid exploitation traps
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.idxname = self.name
        self.agent_scores = torch.tensor([0], dtype=torch.int64)
        self.alpha0 = float(alpha)
        self.forget = float(np.clip(forget, 0.9, 1.0))
        self.eps = float(np.clip(eps, 0.0, 0.2))
        self.counts = np.full(3, self.alpha0, dtype=np.float64)  # Posterior pseudo-counts
        self.last_opp = None
        
        # 批量处理支持（保持兼容性）
        self.batch_size = 1  # 统计模型不需要批量

    def set_batch_size(self, batch_size: int):
        """设置批量大小（对统计模型无效）"""
        pass

    def punches(self, round_idx: Optional[int] = None) -> int:
        # Opponent distribution (posterior mean)
        p = self.counts / self.counts.sum()
        # Predict best response to most likely opponent move
        opp_pred = int(np.argmax(p))
        my = _beat_move(opp_pred)
        if random.random() < self.eps:
            my = random.randint(0, 2)
        return my

    def play(self, my_action: int, opp_action: int) -> None:
        out = _outcome(my_action, opp_action)
        self.agent_scores += _score_delta(out)
        # Exponential forgetting
        self.counts *= self.forget
        self.counts[opp_action] += 1.0
        self.last_opp = opp_action

    def batch_play(self, batch: List[Tuple[int, int]]) -> None:
        """批量更新（直接逐个处理）"""
        for my_action, opp_action in batch:
            self.play(my_action, opp_action)

    def getscores(self):
        return self.agent_scores

    def save(self, idx: Optional[int] = None) -> None:
        # Save counts as state_dict substitute
        state = {
            "counts": self.counts.tolist(),
            "alpha0": self.alpha0,
            "forget": self.forget,
            "eps": self.eps
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
        print(f"{self.idxname} scores: {int(self.agent_scores.item())}")
