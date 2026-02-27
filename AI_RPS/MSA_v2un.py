# -*- coding: utf-8 -*-
"""
MSA_v2.py - 最终修复版多尺度聚合模型

修复记录：
1. ✅ 修复 batch_play 参数兼容性（两个参数）
2. ✅ 统一属性命名 (agent_scores → score)
3. ✅ 修正模型名称 (MSA_Optimized → MSA_v2)
4. ✅ 添加 **kwargs 支持（兼容测试脚本）
5. ✅ 移除 verbose 参数（PyTorch 兼容性）
"""

import math
import random
from typing import Optional, List, Tuple, Dict
from collections import deque

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


def _lose_move(move: int) -> int:
    return (move - 1) % 3


class AdaptiveMSAMixer(nn.Module):
    """改进的混合器，包含注意力机制和自适应权重"""
    def __init__(self, n_feats: int, hidden_dim: int = 16):
        super().__init__()
        self.n_feats = n_feats
        
        # 使用小型神经网络来学习混合权重
        self.fc1 = nn.Linear(n_feats * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_feats)
        self.dropout = nn.Dropout(0.1)
        
        # 温度参数用于控制softmax的锐度
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, feats: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        # feats: [B, n_feats, 3] probabilities per feature
        B, n_feat, C = feats.shape
        
        # 将特征展平并通过网络
        flat_feats = feats.view(B, -1)  # [B, n_feats*3]
        h = F.relu(self.fc1(flat_feats))  # [B, hidden_dim]
        h = self.dropout(h)
        weights_logits = self.fc2(h)  # [B, n_feats]
        
        # 应用温度缩放的softmax
        weights = F.softmax(weights_logits / self.temperature, dim=1)  # [B, n_feats]
        
        # 加权混合分布
        mixed = torch.einsum('bf, bfc -> bc', weights, feats)  # [B, 3]
        
        # 确保输出是有效的概率分布
        mixed = F.softmax(mixed, dim=-1)
        
        if return_weights:
            return mixed, weights
        return mixed


class MetaStrategy:
    """元策略：结合多种策略的预测"""
    def __init__(self, decay_factor: float = 0.99):
        self.strategies = {
            'frequency': self.frequency_strategy,
            'anti_frequency': self.anti_frequency_strategy,
            'rotation': self.rotation_strategy,
            'pattern': self.pattern_strategy,
            'markov': self.markov_strategy
        }
        # 每个策略的累积奖励
        self.strategy_rewards = {k: 0.0 for k in self.strategies}
        self.decay_factor = decay_factor
        self.transition_counts = np.zeros((3, 3))  # Markov transitions
        
    def update_rewards(self, predictions: Dict[str, int], actual: int):
        """更新策略奖励"""
        for name, pred in predictions.items():
            if pred == actual:
                self.strategy_rewards[name] += 1.0
            else:
                self.strategy_rewards[name] -= 0.5
        
        # 应用衰减
        for name in self.strategy_rewards:
            self.strategy_rewards[name] *= self.decay_factor
    
    def frequency_strategy(self, history: List[int]) -> np.ndarray:
        """频率策略：预测最常见的动作"""
        if not history:
            return np.array([1/3, 1/3, 1/3])
        counts = np.bincount(history, minlength=3) + 0.5  # 拉普拉斯平滑
        return counts / counts.sum()
    
    def anti_frequency_strategy(self, history: List[int]) -> np.ndarray:
        """反频率策略：预测最少见的动作"""
        if not history:
            return np.array([1/3, 1/3, 1/3])
        counts = np.bincount(history, minlength=3)
        # 反转频率
        inv_counts = counts.max() - counts + 1
        return inv_counts / inv_counts.sum()
    
    def rotation_strategy(self, history: List[int]) -> np.ndarray:
        """轮转策略：预测下一个动作是当前动作+1"""
        if not history:
            return np.array([1/3, 1/3, 1/3])
        last_move = history[-1]
        probs = np.array([0.1, 0.1, 0.1])
        probs[(last_move + 1) % 3] = 0.7
        return probs
    
    def pattern_strategy(self, history: List[int], lookback: int = 5) -> np.ndarray:
        """模式匹配策略：寻找重复模式"""
        if len(history) < lookback + 1:
            return np.array([1/3, 1/3, 1/3])
        
        # 寻找最近的模式在历史中的匹配
        recent_pattern = history[-lookback:]
        probs = np.array([1/3, 1/3, 1/3])
        
        for i in range(len(history) - lookback - 1):
            if history[i:i+lookback] == recent_pattern:
                next_move = history[i+lookback]
                probs[next_move] += 0.1
        
        return probs / probs.sum()
    
    def markov_strategy(self, history: List[int]) -> np.ndarray:
        """马尔可夫链策略"""
        if len(history) < 2:
            return np.array([1/3, 1/3, 1/3])
        
        # 更新转移计数
        for i in range(len(history) - 1):
            self.transition_counts[history[i], history[i+1]] += 1
        
        last_move = history[-1]
        transitions = self.transition_counts[last_move] + 0.5
        return transitions / transitions.sum()
    
    def get_mixed_prediction(self, history: List[int]) -> np.ndarray:
        """获取混合预测"""
        predictions = {}
        probs_list = []
        weights = []
        
        for name, strategy in self.strategies.items():
            probs = strategy(history)
            predictions[name] = np.argmax(probs)
            probs_list.append(probs)
            
            # 使用softmax转换奖励为权重
            reward = self.strategy_rewards[name]
            weights.append(np.exp(reward / 10.0))  # 温度参数为10
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加权平均
        mixed_probs = sum(w * p for w, p in zip(weights, probs_list))
        return mixed_probs / mixed_probs.sum(), predictions


class Train:
    name = "MSA_v2un"

    def __init__(self, windows=(2, 3, 5, 8, 13, 21), lr: float = 1e-3, 
                 seed: Optional[int] = None, use_meta: bool = True, **kwargs):  # ✅ 添加 **kwargs
        """
        初始化 MSA_v2 模型
        
        Args:
            windows: 多尺度窗口大小
            lr: 学习率
            seed: 随机种子
            use_meta: 是否使用元策略
            **kwargs: 额外参数（兼容测试脚本，如 idxname, name 等）
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)
        self.windows = list(sorted(set(int(w) for w in windows if w >= 2)))
        self.device = _device()
        
        # 使用改进的混合器
        self.mixer = AdaptiveMSAMixer(n_feats=len(self.windows) + 2).to(self.device)  # +2 for global and meta
        
        # 自适应学习率调度器
        self.opt = torch.optim.AdamW(self.mixer.parameters(), lr=lr, weight_decay=1e-4)
        
        # ✅ 修复：移除 verbose 参数（PyTorch 兼容性）
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=50
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.opp_hist: List[int] = []
        self.my_hist: List[int] = []
        
        # 批量处理相关
        self.batch_size = 16
        self.update_buffer: List[Tuple[int, int]] = []
        
        # 元策略
        self.use_meta = use_meta
        self.meta_strategy = MetaStrategy() if use_meta else None
        
        # 性能追踪
        self.recent_losses = deque(maxlen=100)
        self.exploration_rate = 0.15
        
    def set_batch_size(self, batch_size: int):
        """设置批量大小"""
        self.batch_size = max(1, batch_size)

    def _freq_dist(self, seq: List[int], alpha: float = 1.0) -> np.ndarray:
        """计算频率分布，使用狄利克雷先验"""
        counts = np.bincount(seq, minlength=3).astype(np.float64) + alpha
        return counts / counts.sum()

    def _pattern_features(self, history: List[int]) -> np.ndarray:
        """提取模式特征"""
        if len(history) < 3:
            return np.array([1/3, 1/3, 1/3])
        
        # 寻找二阶模式
        pattern_counts = np.zeros(3)
        if len(history) >= 3:
            last_pair = tuple(history[-2:])
            for i in range(len(history) - 2):
                if tuple(history[i:i+2]) == last_pair:
                    pattern_counts[history[i+2]] += 1
        
        if pattern_counts.sum() > 0:
            return (pattern_counts + 0.5) / (pattern_counts.sum() + 1.5)
        return np.array([1/3, 1/3, 1/3])

    def _feats(self, history: Optional[List[int]] = None) -> torch.Tensor:
        """构建增强特征"""
        if history is None:
            history = self.opp_hist
        
        if not history:
            # 初始均匀分布
            feats = np.tile(np.array([1/3, 1/3, 1/3], dtype=np.float64), 
                           (len(self.windows) + 2, 1))
        else:
            # 全局频率
            glb = self._freq_dist(history, alpha=1.0)
            
            # 不同窗口的频率
            windows = []
            for i, w in enumerate(self.windows):
                if len(history) >= w:
                    alpha = 1.0 / (1 + i * 0.5)
                    windows.append(self._freq_dist(history[-w:], alpha=alpha))
                else:
                    noise = np.random.dirichlet([1, 1, 1]) * 0.1
                    windows.append(glb * 0.9 + noise)
            
            # 添加元策略特征
            if self.use_meta and self.meta_strategy:
                meta_probs, _ = self.meta_strategy.get_mixed_prediction(history)
            else:
                meta_probs = self._pattern_features(history)
            
            feats = np.vstack([glb] + windows + [meta_probs])
        
        t = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        return t

    def punches(self, round_idx: Optional[int] = None) -> int:
        """决策函数"""
        # 探索vs利用
        if random.random() < self.exploration_rate:
            return random.randint(0, 2)
        
        with torch.no_grad():
            dist = self.mixer(self._feats()).squeeze(0)
            
            # 早期添加噪声
            if len(self.opp_hist) < 50:
                noise = torch.randn(3, device=self.device) * 0.1
                dist = F.softmax(torch.log(dist + 1e-8) + noise, dim=-1)
            
            # 20%概率采样
            if random.random() < 0.2:
                opp_pred = torch.multinomial(dist, 1).item()
            else:
                opp_pred = int(torch.argmax(dist).item())
        
        return _beat_move(opp_pred)

    def play(self, my_action: int, opp_action: int) -> None:
        """单个更新"""
        out = _outcome(my_action, opp_action)
        self.score += _score_delta(out)
        
        # 更新元策略
        if self.use_meta and self.meta_strategy and len(self.opp_hist) > 0:
            _, predictions = self.meta_strategy.get_mixed_prediction(self.opp_hist)
            self.meta_strategy.update_rewards(predictions, opp_action)
        
        # 添加到缓存
        self.update_buffer.append((my_action, opp_action))
        self.opp_hist.append(opp_action)
        self.my_hist.append(my_action)
        
        # 动态调整探索率
        if len(self.opp_hist) % 10 == 0:
            self.exploration_rate *= 0.995
            self.exploration_rate = max(0.05, self.exploration_rate)
        
        # 批量更新
        if len(self.update_buffer) >= self.batch_size:
            self._batch_update()

    def batch_play(self, my_actions, opp_actions):
        """批量更新（兼容接口）"""
        for my_action, opp_action in zip(my_actions, opp_actions):
            out = _outcome(int(my_action), int(opp_action))
            self.score += _score_delta(out)
            self.opp_hist.append(int(opp_action))
            self.my_hist.append(int(my_action))
            
            # 更新元策略
            if self.use_meta and self.meta_strategy and len(self.opp_hist) > 1:
                _, predictions = self.meta_strategy.get_mixed_prediction(self.opp_hist[:-1])
                self.meta_strategy.update_rewards(predictions, int(opp_action))
        
        # 批量训练
        if len(self.opp_hist) >= 2:
            batch = list(zip(my_actions, opp_actions))
            self._batch_train(batch)

    def _batch_update(self):
        """执行批量更新"""
        if not self.update_buffer:
            return
        
        if len(self.opp_hist) >= 2:
            self._batch_train(self.update_buffer)
        
        self.update_buffer.clear()

    def _batch_train(self, batch: List[Tuple[int, int]]):
        """批量训练"""
        if len(batch) == 0 or len(self.opp_hist) < 5:
            return
        
        # 准备训练数据
        batch_inputs = []
        batch_targets = []
        
        max_samples = min(len(self.opp_hist) - 1, self.batch_size * 2)
        sample_indices = np.random.choice(
            range(1, len(self.opp_hist)), 
            size=min(max_samples, len(self.opp_hist) - 1),
            replace=False
        )
        
        for idx in sample_indices:
            hist_slice = self.opp_hist[:idx]
            if len(hist_slice) >= 2:
                feats = self._feats(hist_slice)
                batch_inputs.append(feats.squeeze(0))
                batch_targets.append(self.opp_hist[idx])
        
        if len(batch_inputs) == 0:
            return
        
        # 前向传播
        inputs = torch.stack(batch_inputs)
        targets = torch.tensor(batch_targets, dtype=torch.long, device=self.device)
        preds = self.mixer(inputs)
        
        # 计算损失
        loss = -torch.mean(torch.log(preds[range(len(targets)), targets] + 1e-8))
        l2_reg = sum(p.pow(2).sum() for p in self.mixer.parameters())
        loss = loss + 1e-5 * l2_reg
        
        self.recent_losses.append(loss.item())
        
        # 反向传播
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), max_norm=1.0)
        self.opt.step()
        
        # 更新学习率
        if len(self.recent_losses) == self.recent_losses.maxlen:
            avg_loss = np.mean(self.recent_losses)
            self.scheduler.step(avg_loss)

    def getscores(self):
        return self.score

    def save(self, idx: Optional[int] = None) -> None:
        self._batch_update()
        
        import os
        os.makedirs("models", exist_ok=True)
        
        state = {
            "mixer": self.mixer.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "windows": self.windows,
            "exploration_rate": self.exploration_rate,
            "meta_rewards": self.meta_strategy.strategy_rewards if self.meta_strategy else None
        }
        torch.save(state, f"models/{self.idxname}_agent.pth")
        print(f"{self.idxname} scores: {int(self.score.item())}")
        
        if self.meta_strategy:
            print(f"Meta strategy rewards: {self.meta_strategy.strategy_rewards}")
        if self.recent_losses:
            print(f"Recent avg loss: {np.mean(self.recent_losses):.4f}")