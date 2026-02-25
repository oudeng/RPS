# -*- coding: utf-8 -*-
"""
Tr.py - GPU优化版Transformer模型（支持批量处理）
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


class TinyTransformer(nn.Module):
    def __init__(self, d_model=48, nhead=4, num_layers=2, max_len=32):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x_idx):
        # x_idx: [B, T] ints in {0,1,2}
        T = x_idx.size(1)
        x = self.emb(x_idx) + self.pos[:, :T, :]
        h = self.encoder(x)
        return self.fc(h[:, -1, :])


class Train:
    name = "Tr_v1"

    def __init__(self, ctx_len: int = 16, lr: float = 1e-3, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.idxname = self.name
        self.agent_scores = torch.tensor([0], dtype=torch.int64)
        self.ctx_len = int(max(4, ctx_len))
        self.device = _device()
        self.net = TinyTransformer(max_len=self.ctx_len).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opp_hist: List[int] = []
        
        # 批量处理相关
        self.batch_size = 32
        self.update_buffer: List[Tuple[int, int]] = []
        
    def set_batch_size(self, batch_size: int):
        """设置批量大小"""
        self.batch_size = max(1, batch_size)

    def punches(self, round_idx: Optional[int] = None) -> int:
        if len(self.opp_hist) < 2:
            return random.randint(0, 2)
        
        ctx = self.opp_hist[-self.ctx_len:]
        x_idx = torch.tensor([ctx], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = self.net(x_idx)
            opp_pred = int(torch.argmax(logits, dim=1).item())
        
        return _beat_move(opp_pred)

    def play(self, my_action: int, opp_action: int) -> None:
        """单个更新（兼容接口）"""
        out = _outcome(my_action, opp_action)
        self.agent_scores += _score_delta(out)
        
        # 添加到缓存
        self.update_buffer.append((my_action, opp_action))
        self.opp_hist.append(opp_action)
        
        # 如果缓存满了，执行批量更新
        if len(self.update_buffer) >= self.batch_size:
            self._batch_update()

    def batch_play(self, batch: List[Tuple[int, int]]) -> None:
        """批量更新（优化接口）"""
        # 更新分数和历史
        for my_action, opp_action in batch:
            out = _outcome(my_action, opp_action)
            self.agent_scores += _score_delta(out)
            self.opp_hist.append(opp_action)
        
        # 批量训练
        if len(self.opp_hist) >= self.ctx_len + 1:
            self._batch_train(batch)

    def _batch_update(self):
        """执行批量更新"""
        if not self.update_buffer:
            return
        
        if len(self.opp_hist) >= self.ctx_len + 1:
            self._batch_train(self.update_buffer)
        
        self.update_buffer.clear()

    def _batch_train(self, batch: List[Tuple[int, int]]):
        """批量训练网络"""
        if not batch:
            return
        
        # 准备批量数据
        batch_inputs = []
        batch_targets = []
        
        # 回溯到批次开始前的历史
        temp_hist = self.opp_hist[:-len(batch)] if len(self.opp_hist) > len(batch) else []
        
        for my_action, opp_action in batch:
            if len(temp_hist) >= self.ctx_len:
                ctx = temp_hist[-self.ctx_len:]
                batch_inputs.append(ctx)
                batch_targets.append(opp_action)
            temp_hist.append(opp_action)
        
        if batch_inputs:
            # 批量前向传播
            # 将所有上下文转换为张量
            max_len = max(len(ctx) for ctx in batch_inputs)
            padded_inputs = []
            
            for ctx in batch_inputs:
                # 填充到相同长度
                padded = ctx + [0] * (max_len - len(ctx))
                padded_inputs.append(padded)
            
            inputs = torch.tensor(padded_inputs, dtype=torch.long, device=self.device)
            targets = torch.tensor(batch_targets, dtype=torch.long, device=self.device)
            
            # 前向传播
            logits = self.net(inputs)
            loss = self.loss_fn(logits, targets)
            
            # 批量反向传播
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()

    def getscores(self):
        return self.agent_scores

    def save(self, idx: Optional[int] = None) -> None:
        # 确保所有缓存都已处理
        self._batch_update()
        
        state = {"model": self.net.state_dict(), "ctx_len": self.ctx_len}
        torch.save(state, f"models/{self.idxname}_agent.pth")
        print(f"{self.idxname} scores: {int(self.agent_scores.item())}")
