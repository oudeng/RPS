# -*- coding: utf-8 -*-
"""
WL.py - Win-Stay/Lose-Shift策略（CPU版）
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
    name = "WL"

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.idxname = self.name
        self.agent_scores = torch.tensor([0], dtype=torch.int64)
        self._next = random.randint(0, 2)
        self._last_my = None
        self._last_opp = None
        
        # 批量处理支持（保持兼容性）
        self.batch_size = 1  # 简单策略不需要批量

    def set_batch_size(self, batch_size: int):
        """设置批量大小（对简单策略无效）"""
        pass

    def punches(self, round_idx: Optional[int] = None) -> int:
        return self._next

    def play(self, my_action: int, opp_action: int) -> None:
        out = _outcome(my_action, opp_action)
        self.agent_scores += _score_delta(out)
        # Update next policy
        if out == 2:  # Win -> stay
            self._next = my_action
        elif out == 1:  # Lose -> shift to beat opponent
            self._next = _beat_move(opp_action)
        else:  # Tie
            self._next = random.randint(0, 2)
        self._last_my, self._last_opp = my_action, opp_action

    def batch_play(self, batch: List[Tuple[int, int]]) -> None:
        """批量更新（直接逐个处理）"""
        for my_action, opp_action in batch:
            self.play(my_action, opp_action)

    def getscores(self):
        return self.agent_scores

    def save(self, idx: Optional[int] = None) -> None:
        print(f"{self.idxname} scores: {int(self.agent_scores.item())}")
