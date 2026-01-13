# -*- coding: utf-8 -*-
"""
MSA.py - 简化优化版多尺度聚合模型

要点：
- 添加模式记忆机制
- 使用KL散度损失（更适合概率分布）
- 温度参数控制早期探索

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


class SimpleMSAMixer(nn.Module):
    """简化但有效的混合器"""
    def __init__(self, n_feats: int):
        super().__init__()
        # 使用可学习的权重，但初始化为均匀分布
        self.w = nn.Parameter(torch.ones(n_feats) / n_feats)
        # 添加偏置项以增加灵活性
        self.bias = nn.Parameter(torch.zeros(3))
        
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, n_feats, 3] probabilities per feature
        B, n_feat, C = feats.shape
        
        # 使用softmax确保权重和为1
        weights = F.softmax(self.w.view(1, n_feat), dim=1)  # [1, n_feat]
        
        # 加权混合
        mixed = torch.einsum('bf, bfc -> bc', weights.expand(B, n_feat), feats)  # [B, 3]
        
        # 添加偏置并归一化
        mixed = mixed + self.bias
        mixed = F.softmax(mixed, dim=-1)
        
        return mixed


class Train:
    name = "MSA_v1"

    def __init__(self, windows=(3, 5, 10, 20, 50), lr: float = 2e-3, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.idxname = self.name
        self.agent_scores = torch.tensor([0], dtype=torch.int64)
        self.windows = list(sorted(set(int(w) for w in windows if w >= 2)))
        self.device = _device()
        
        # 核心改进1：更多特征
        # +3 for: global, recent(last 2), pattern
        self.mixer = SimpleMSAMixer(n_feats=len(self.windows) + 3).to(self.device)
        
        # 核心改进2：更好的优化器设置
        self.opt = torch.optim.Adam(self.mixer.parameters(), lr=lr, weight_decay=5e-5)
        
        # 使用KL散度损失，更适合概率分布
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        self.opp_hist: List[int] = []
        self.my_hist: List[int] = []
        
        # 批量处理
        self.batch_size = 8  # 更小的批量
        self.update_buffer: List[Tuple[int, int]] = []
        
        # 核心改进3：跟踪对手模式
        self.pattern_memory = {}  # 存储模式->下一步的映射
        self.max_pattern_len = 3
        
    def set_batch_size(self, batch_size: int):
        """设置批量大小"""
        self.batch_size = max(1, batch_size)

    def _freq_dist(self, seq: List[int], alpha: float = 0.5) -> np.ndarray:
        """计算频率分布，使用较小的平滑参数"""
        if not seq:
            return np.array([1/3, 1/3, 1/3])
        counts = np.bincount(seq, minlength=3).astype(np.float64) + alpha
        return counts / counts.sum()
    
    def _update_pattern_memory(self):
        """更新模式记忆"""
        if len(self.opp_hist) < self.max_pattern_len + 1:
            return
        
        # 提取最近的模式
        for pattern_len in range(2, self.max_pattern_len + 1):
            if len(self.opp_hist) >= pattern_len + 1:
                pattern = tuple(self.opp_hist[-pattern_len-1:-1])
                next_move = self.opp_hist[-1]
                
                if pattern not in self.pattern_memory:
                    self.pattern_memory[pattern] = [0, 0, 0]
                self.pattern_memory[pattern][next_move] += 1
    
    def _get_pattern_prediction(self) -> np.ndarray:
        """基于模式记忆预测"""
        if len(self.opp_hist) < 2:
            return np.array([1/3, 1/3, 1/3])
        
        predictions = np.array([1.0, 1.0, 1.0])  # 使用1作为基础计数
        
        # 检查不同长度的模式
        for pattern_len in range(2, min(len(self.opp_hist), self.max_pattern_len + 1)):
            pattern = tuple(self.opp_hist[-pattern_len:])
            if pattern in self.pattern_memory:
                counts = self.pattern_memory[pattern]
                predictions += np.array(counts)
        
        return predictions / predictions.sum()

    def _feats(self, history: Optional[List[int]] = None) -> torch.Tensor:
        """构建特征，包含多尺度和模式信息"""
        if history is None:
            history = self.opp_hist
        
        if not history:
            # 初始均匀分布
            n_feats = len(self.windows) + 3
            feats = np.tile(np.array([1/3, 1/3, 1/3], dtype=np.float64), (n_feats, 1))
        else:
            feats_list = []
            
            # 1. 全局频率
            feats_list.append(self._freq_dist(history, alpha=0.5))
            
            # 2. 多尺度窗口频率
            for w in self.windows:
                if len(history) >= w:
                    feats_list.append(self._freq_dist(history[-w:], alpha=0.3))
                else:
                    # 使用部分历史
                    feats_list.append(self._freq_dist(history, alpha=1.0))
            
            # 3. 最近动作（last 2）
            if len(history) >= 2:
                recent = self._freq_dist(history[-2:], alpha=0.1)
            else:
                recent = self._freq_dist(history, alpha=0.5)
            feats_list.append(recent)
            
            # 4. 模式预测
            pattern_pred = self._get_pattern_prediction()
            feats_list.append(pattern_pred)
            
            feats = np.vstack(feats_list)
        
        t = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        return t

    def punches(self, round_idx: Optional[int] = None) -> int:
        """决策函数"""
        with torch.no_grad():
            dist = self.mixer(self._feats()).squeeze(0)  # [3]
            
            # 早期增加随机性
            if len(self.opp_hist) < 20:
                # 使用温度参数增加随机性
                temperature = 2.0 - (len(self.opp_hist) / 20.0)  # 从2.0降到1.0
                dist = F.softmax(torch.log(dist + 1e-8) / temperature, dim=-1)
                # 采样而不是取最大值
                opp_pred = torch.multinomial(dist, 1).item()
            else:
                # 后期更确定性
                opp_pred = int(torch.argmax(dist).item())
        
        return _beat_move(opp_pred)

    def play(self, my_action: int, opp_action: int) -> None:
        """单个更新"""
        out = _outcome(my_action, opp_action)
        self.agent_scores += _score_delta(out)
        
        # 更新历史
        self.opp_hist.append(opp_action)
        self.my_hist.append(my_action)
        
        # 更新模式记忆
        self._update_pattern_memory()
        
        # 添加到缓存
        self.update_buffer.append((my_action, opp_action))
        
        # 批量更新
        if len(self.update_buffer) >= self.batch_size:
            self._batch_update()

    def batch_play(self, batch: List[Tuple[int, int]]) -> None:
        """批量更新"""
        for my_action, opp_action in batch:
            out = _outcome(my_action, opp_action)
            self.agent_scores += _score_delta(out)
            self.opp_hist.append(opp_action)
            self.my_hist.append(my_action)
            self._update_pattern_memory()
        
        # 批量训练
        if len(self.opp_hist) >= 3:
            self._batch_train(batch)

    def _batch_update(self):
        """执行批量更新"""
        if not self.update_buffer or len(self.opp_hist) < 3:
            return
        
        self._batch_train(self.update_buffer)
        self.update_buffer.clear()

    def _batch_train(self, batch: List[Tuple[int, int]]):
        """简化但有效的批量训练"""
        if len(batch) == 0 or len(self.opp_hist) < 3:
            return
        
        # 创建训练样本
        batch_inputs = []
        batch_targets = []
        
        # 使用历史的不同点作为训练数据
        # 确保我们有足够的历史
        min_hist_len = min(3, self.windows[0] if self.windows else 3)
        
        if len(self.opp_hist) > min_hist_len:
            # 随机采样历史点
            n_samples = min(len(batch) * 2, len(self.opp_hist) - 1)
            
            for _ in range(n_samples):
                # 随机选择一个历史点
                idx = random.randint(min_hist_len, len(self.opp_hist) - 1)
                hist_slice = self.opp_hist[:idx]
                
                # 创建特征和目标
                feats = self._feats(hist_slice)
                batch_inputs.append(feats.squeeze(0))
                batch_targets.append(self.opp_hist[idx])
        
        if not batch_inputs:
            return
        
        # 批量前向传播
        inputs = torch.stack(batch_inputs)  # [B, n_feats, 3]
        targets = torch.tensor(batch_targets, dtype=torch.long, device=self.device)
        
        # 获取预测分布
        preds = self.mixer(inputs)  # [B, 3]
        
        # 创建目标分布（one-hot）
        target_dist = torch.zeros_like(preds)
        target_dist.scatter_(1, targets.unsqueeze(1), 1)
        
        # KL散度损失
        log_preds = torch.log(preds + 1e-8)
        loss = self.loss_fn(log_preds, target_dist)
        
        # 反向传播
        self.opt.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), max_norm=1.0)
        
        self.opt.step()

    def getscores(self):
        return self.agent_scores

    def save(self, idx: Optional[int] = None) -> None:
        # 确保所有缓存都已处理
        self._batch_update()
        
        state = {
            "mixer": self.mixer.state_dict(),
            "windows": self.windows,
            "pattern_memory": self.pattern_memory
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
        print(f"{self.idxname} scores: {int(self.agent_scores.item())}")
        
        # 调试信息
        if len(self.pattern_memory) > 0:
            print(f"Pattern memory size: {len(self.pattern_memory)}")