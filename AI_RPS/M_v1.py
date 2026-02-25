# -*- coding: utf-8 -*-
"""
M_v1.py - 一阶马尔可夫模型（CPU优化版）
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
    name = "M_v1"

    def __init__(self, decay: float = 0.98, smooth: float = 0.5,
                 update_on_win_only: bool = False, seed: Optional[int] = None):
        """
        First-order Markov model of opponent moves:
        - decay: exponential decay applied to transition counts
        - smooth: additive smoothing
        - update_on_win_only: if True, only update when this agent wins
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.idxname = self.name
        self.agent_scores = torch.tensor([0], dtype=torch.int64)
        self.decay = float(np.clip(decay, 0.9, 1.0))
        self.smooth = float(np.clip(smooth, 0.0, 5.0))
        self.update_on_win_only = bool(update_on_win_only)
        self.T = np.full((3, 3), self.smooth, dtype=np.float64)  # Rows=prev opp move, cols=next opp move
        self.last_opp = None
        
        # 批量处理支持（保持兼容性）
        self.batch_size = 1  # 统计模型不需要批量

    def set_batch_size(self, batch_size: int):
        """设置批量大小（对统计模型无效）"""
        pass

    def _predict_opp(self) -> int:
        if self.last_opp is None:
            # Fall back to stationary distribution
            row = self.T.sum(axis=0)
        else:
            row = self.T[self.last_opp]
        if row.sum() <= 0:
            probs = np.array([1/3, 1/3, 1/3], dtype=np.float64)
        else:
            probs = row / row.sum()
        return int(np.argmax(probs))

    def punches(self, round_idx: Optional[int] = None) -> int:
        opp_pred = self._predict_opp()
        return _beat_move(opp_pred)

    def play(self, my_action: int, opp_action: int) -> None:
        out = _outcome(my_action, opp_action)
        self.agent_scores += _score_delta(out)
        do_upd = True
        if self.update_on_win_only and out != 2:
            do_upd = False
        if do_upd and self.last_opp is not None:
            self.T *= self.decay
            self.T[self.last_opp, opp_action] += 1.0
        elif do_upd and self.last_opp is None:
            # Only stationary counts at the very first observation
            self.T[:, opp_action] += 1.0 * (1.0 / 3.0)  # Distribute lightly
        self.last_opp = opp_action

    def batch_play(self, batch: List[Tuple[int, int]]) -> None:
        """批量更新（直接逐个处理）"""
        for my_action, opp_action in batch:
            self.play(my_action, opp_action)

    def getscores(self):
        return self.agent_scores

    def save(self, idx: Optional[int] = None) -> None:
        state = {
            "T": self.T.tolist(),
            "decay": self.decay,
            "smooth": self.smooth,
            "update_on_win_only": self.update_on_win_only
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
        print(f"{self.idxname} scores: {int(self.agent_scores.item())}")
