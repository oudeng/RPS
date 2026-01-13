# -*- coding: utf-8 -*-
"""
RPS agent implementation with last_policy tracking for Lipschitz analysis.
Modified to record the actual policy distribution used for decision making.
"""

from typing import List, Optional
import random
import math
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

def _freq_pred(hist: List[int]) -> int:
    if not hist:
        return random.randint(0,2)
    counts = np.bincount(hist, minlength=3)
    return int(np.argmax(counts))

import torch.nn as nn
import torch.nn.functional as F

def _device():
    # Prefer GPU for neural net
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUNet(nn.Module):
    def __init__(self, input_size=3, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden, 3)
    def forward(self, x):
        # x: [B, T, 3]
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.fc(h)

class Train:
    name = "RNN_v1"
    def __init__(self, ctx_len: int=16, lr: float=1e-3, hidden:int=64, layers:int=1, dropout:float=0.0, seed: Optional[int]=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)
        self.ctx_len = int(max(4, ctx_len))
        self.device = _device()
        self.net = GRUNet(input_size=3, hidden=hidden, num_layers=layers, dropout=dropout).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opp_hist: List[int] = []
        self._batch_size = 1  # optional for compat
        
        # NEW: Track last policy distribution for Lipschitz analysis
        self.last_policy = None

    # optional batch interface for training harness compatibility
    def set_batch_size(self, bs: int):
        self._batch_size = max(1, int(bs))

    def _one_hot_seq(self, seq: List[int]) -> torch.Tensor:
        T = len(seq)
        x = torch.zeros((1, T, 3), dtype=torch.float32, device=self.device)
        for i, a in enumerate(seq):
            x[0, i, a] = 1.0
        return x

    def punches(self, round_idx: Optional[int]=None) -> int:
        if len(self.opp_hist) < 2:
            # NEW: Record uniform policy for initial rounds
            self.last_policy = np.array([1/3, 1/3, 1/3])
            return random.randint(0,2)
        
        ctx = self.opp_hist[-self.ctx_len:]
        x = self._one_hot_seq(ctx)
        with torch.no_grad():
            logits = self.net(x)
            # NEW: Extract and record the policy distribution
            probs = F.softmax(logits, dim=-1)
            self.last_policy = probs.squeeze().cpu().numpy()
            
            # Original decision logic (argmax to predict opponent, then beat)
            opp_pred = int(torch.argmax(logits, dim=1).item())
        return _beat(opp_pred)

    def _train_step(self, ctx_seq: List[int], target: int):
        x = self._one_hot_seq(ctx_seq)
        y = torch.tensor([target], dtype=torch.long, device=self.device)
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

    def play(self, my_action: int, opp_action: int) -> None:
        out = _outcome(my_action, opp_action)
        self.score += _score_delta(out)
        # train to predict current opp_action from previous context
        if len(self.opp_hist) >= 1:
            ctx = self.opp_hist[-self.ctx_len:]
            self._train_step(ctx, opp_action)
        self.opp_hist.append(opp_action)

    def batch_play(self, my_actions, opp_actions):
        for a_my, a_opp in zip(my_actions, opp_actions):
            self.play(int(a_my), int(a_opp))

    def getscores(self):
        return self.score

    def save(self, idx: Optional[int]=None) -> None:
        import os
        os.makedirs("models", exist_ok=True)
        state = {"model": self.net.state_dict(), "ctx_len": self.ctx_len}
        torch.save(state, f"models/{self.idxname}_agent.pth")
        
    def load(self) -> None:
        """Load saved model"""
        import os
        path = f"models/{self.idxname}_agent.pth"
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.net.load_state_dict(state["model"])
            if "ctx_len" in state:
                self.ctx_len = state["ctx_len"]