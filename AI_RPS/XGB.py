# -*- coding: utf-8 -*-
"""
XGB_optimized.py - 优化版 XGBoost 模型

主要改进：
1. 修复内存泄漏：限制 X, y 列表最大长度
2. 优化转移概率计算：增量更新而非重新计算
3. 添加样本权重：近期样本权重更高
4. 改进 Early Stopping API 兼容性
5. 缓存特征提取结果，提升效率
"""

from typing import List, Optional, Tuple
import random
import math
import numpy as np
import torch
from collections import deque

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
        return random.randint(0, 2)
    counts = np.bincount(hist, minlength=3)
    counts = counts + np.random.random(3) * 0.1
    return int(np.argmax(counts))

try:
    import xgboost as xgb
    _XGB_OK = True
    # 检查 XGBoost 版本
    _XGB_VERSION = tuple(map(int, xgb.__version__.split('.')[:2]))
except Exception:
    _XGB_OK = False
    _XGB_VERSION = (0, 0)


class IncrementalStats:
    """增量统计工具，避免每次重新计算"""
    def __init__(self):
        self.trans_matrix = np.zeros((3, 3), dtype=np.float32)  # 一阶转移
        self.trans2_matrix = np.zeros((3, 3, 3), dtype=np.float32)  # 二阶转移
        self.battle_matrix = np.zeros((3, 3), dtype=np.float32)  # 对战统计
        self.total_trans = 0
        self.total_trans2 = 0
        self.total_battle = 0
    
    def update_transition(self, prev: int, curr: int):
        """更新一阶转移"""
        self.trans_matrix[prev, curr] += 1
        self.total_trans += 1
    
    def update_transition2(self, prev2: int, prev1: int, curr: int):
        """更新二阶转移"""
        self.trans2_matrix[prev2, prev1, curr] += 1
        self.total_trans2 += 1
    
    def update_battle(self, my: int, opp: int):
        """更新对战统计"""
        self.battle_matrix[my, opp] += 1
        self.total_battle += 1
    
    def get_trans_probs(self) -> np.ndarray:
        """获取归一化的转移概率"""
        if self.total_trans > 0:
            return (self.trans_matrix / self.total_trans).reshape(-1)
        return np.zeros(9, dtype=np.float32)
    
    def get_trans2_probs(self) -> np.ndarray:
        """获取归一化的二阶转移概率"""
        if self.total_trans2 > 0:
            return (self.trans2_matrix / self.total_trans2).reshape(-1)
        return np.zeros(27, dtype=np.float32)
    
    def get_battle_probs(self) -> np.ndarray:
        """获取归一化的对战概率"""
        if self.total_battle > 0:
            return (self.battle_matrix / self.total_battle).reshape(-1)
        return np.ones(9, dtype=np.float32) / 9


def _enhanced_features_fast(
    opp_hist: List[int], 
    my_hist: List[int], 
    stats: IncrementalStats,
    ctx_len: int = 16
) -> np.ndarray:
    """优化的特征提取（使用增量统计）"""
    features = []
    
    # 1. One-hot 编码的对手历史（右对齐）
    opp_tail = opp_hist[-ctx_len:] if len(opp_hist) >= ctx_len else opp_hist
    opp_one_hot = np.zeros((ctx_len, 3), dtype=np.float32)
    start_idx = ctx_len - len(opp_tail)
    for i, a in enumerate(opp_tail):
        opp_one_hot[start_idx + i, a] = 1.0
    features.append(opp_one_hot.reshape(-1))
    
    # 2. 全局频率
    if opp_hist:
        opp_freq = np.bincount(opp_hist, minlength=3).astype(np.float32)
        opp_freq = opp_freq / (opp_freq.sum() + 1e-5)
    else:
        opp_freq = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    features.append(opp_freq)
    
    # 3. 多尺度频率
    for w in [5, 10, 20, 50]:
        if len(opp_hist) >= w:
            recent_freq = np.bincount(opp_hist[-w:], minlength=3).astype(np.float32)
            recent_freq = recent_freq / (recent_freq.sum() + 1e-5)
        else:
            recent_freq = opp_freq
        features.append(recent_freq)
    
    # 4. 转移概率（使用增量统计）
    features.append(stats.get_trans_probs())
    
    # 5. 二阶转移概率（使用增量统计）
    features.append(stats.get_trans2_probs())
    
    # 6. 对战统计（使用增量统计）
    features.append(stats.get_battle_probs())
    
    # 7. 连续相同动作计数
    streak_features = np.zeros(3, dtype=np.float32)
    if opp_hist:
        last_action = opp_hist[-1]
        streak = 1
        for i in range(len(opp_hist) - 2, -1, -1):
            if opp_hist[i] == last_action:
                streak += 1
            else:
                break
        streak_features[last_action] = min(streak / 10.0, 1.0)
    features.append(streak_features)
    
    return np.concatenate(features, axis=0)


class Train:
    name = "XGB"
    
    def __init__(
        self,
        ctx_len: int = 16,
        n_estimators: int = 150,
        max_depth: int = 5,  # 降低深度防止过拟合
        lr: float = 0.08,
        subsample: float = 0.8,
        colsample: float = 0.8,
        retrain_every: int = 40,
        min_fit: int = 60,
        max_samples: int = 3000,
        use_gpu: bool = False,
        use_sample_weight: bool = True,  # 新增：是否使用样本权重
        seed: Optional[int] = None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.idxname = self.name
        self.score = torch.tensor([0], dtype=torch.int64)
        self.ctx_len = int(max(4, ctx_len))
        self.retrain_every = int(max(20, retrain_every))
        self.min_fit = int(max(40, min_fit))
        self.max_samples = int(max(500, max_samples))
        self.use_sample_weight = use_sample_weight
        
        # 使用 deque 限制最大长度，自动删除旧数据
        self.opp_hist: List[int] = []
        self.my_hist: List[int] = []
        self.X = deque(maxlen=self.max_samples * 2)  # 保留 2 倍容量作为缓冲
        self.y = deque(maxlen=self.max_samples * 2)
        
        self.last_fit_n = 0
        self.model = None
        self.rounds_played = 0
        
        # 增量统计
        self.stats = IncrementalStats()
        
        # GPU 配置
        self.use_gpu = bool(use_gpu) and _XGB_OK and torch.cuda.is_available()
        
        if _XGB_OK:
            self.model = xgb.XGBClassifier(
                objective="multi:softprob",
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=subsample,
                colsample_bytree=colsample,
                min_child_weight=3,  # 增加以防止过拟合
                reg_alpha=0.05,  # 增强正则化
                reg_lambda=0.1,
                tree_method="gpu_hist" if self.use_gpu else "hist",
                predictor="gpu_predictor" if self.use_gpu else "cpu_predictor",
                random_state=seed,
                n_jobs=-1 if not self.use_gpu else 1,
                verbosity=0,
                eval_metric='mlogloss'
            )
    
    def punches(self, round_idx: Optional[int] = None) -> int:
        """预测对手动作并返回克制动作"""
        # 早期探索
        if self.rounds_played < 30:
            if random.random() < 0.2:
                return random.randint(0, 2)
        
        if _XGB_OK and self.model is not None and hasattr(self.model, "classes_"):
            try:
                f = _enhanced_features_fast(
                    self.opp_hist, self.my_hist, self.stats, self.ctx_len
                ).reshape(1, -1)
                
                # 使用概率预测
                probs = self.model.predict_proba(f)[0]
                
                # 早期添加噪声
                if self.rounds_played < 100:
                    noise = np.random.random(3) * 0.1
                    probs = probs + noise
                    probs = probs / probs.sum()
                
                # 5% 概率随机采样（探索）
                if random.random() < 0.05:
                    opp_pred = np.random.choice(3, p=probs)
                else:
                    opp_pred = int(np.argmax(probs))
            
            except Exception as e:
                # 降级到频率预测
                opp_pred = _freq_pred(self.opp_hist)
        else:
            opp_pred = _freq_pred(self.opp_hist)
        
        return _beat(opp_pred)
    
    def _compute_sample_weights(self, n: int) -> np.ndarray:
        """计算样本权重：近期样本权重更高（指数衰减）"""
        if not self.use_sample_weight:
            return np.ones(n, dtype=np.float32)
        
        # 指数衰减：最旧样本权重 ~0.37，最新样本权重 1.0
        weights = np.exp(np.linspace(-1, 0, n))
        return weights.astype(np.float32)
    
    def _maybe_fit(self):
        """训练模型"""
        if not (_XGB_OK and self.model is not None):
            return
        
        n = len(self.y)
        
        # 动态调整训练频率
        retrain_interval = self.retrain_every
        if n > 500:
            retrain_interval = int(self.retrain_every * 1.5)
        if n > 1000:
            retrain_interval = int(self.retrain_every * 2)
        
        if n >= self.min_fit and (n - self.last_fit_n) >= retrain_interval:
            # 转换为 numpy 数组（取最后 max_samples 个）
            X = np.array(list(self.X)[-self.max_samples:], dtype=np.float32)
            y = np.array(list(self.y)[-self.max_samples:], dtype=np.int64)
            
            # 计算样本权重
            weights = self._compute_sample_weights(len(X))
            
            # 验证集
            if len(X) > 200:
                val_size = min(100, len(X) // 5)
                X_train, X_val = X[:-val_size], X[-val_size:]
                y_train, y_val = y[:-val_size], y[-val_size:]
                w_train, w_val = weights[:-val_size], weights[-val_size:]
                
                try:
                    # 兼容新旧版本 XGBoost
                    if _XGB_VERSION >= (2, 0):
                        # XGBoost 2.0+ 使用回调
                        self.model.fit(
                            X_train, y_train,
                            sample_weight=w_train,
                            eval_set=[(X_val, y_val)],
                            sample_weight_eval_set=[w_val],
                            callbacks=[xgb.callback.EarlyStopping(rounds=10)],
                            verbose=False
                        )
                    else:
                        # XGBoost 1.x 使用旧 API
                        self.model.fit(
                            X_train, y_train,
                            sample_weight=w_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=10,
                            verbose=False
                        )
                    self.last_fit_n = n
                
                except Exception:
                    # 降级：不使用验证集
                    try:
                        self.model.fit(X, y, sample_weight=weights, verbose=False)
                        self.last_fit_n = n
                    except:
                        pass
            else:
                # 样本少时直接训练
                try:
                    self.model.fit(X, y, sample_weight=weights, verbose=False)
                    self.last_fit_n = n
                except Exception:
                    pass
    
    def play(self, my_action: int, opp_action: int) -> None:
        """单步更新"""
        # 更新分数
        out = _outcome(my_action, opp_action)
        self.score += _score_delta(out)
        
        # 提取特征（使用当前历史，不包含当前动作）
        f = _enhanced_features_fast(
            self.opp_hist, self.my_hist, self.stats, self.ctx_len
        )
        
        # 保存样本
        self.X.append(f)
        self.y.append(opp_action)
        
        # 增量更新统计信息
        if len(self.opp_hist) >= 1:
            self.stats.update_transition(self.opp_hist[-1], opp_action)
        if len(self.opp_hist) >= 2:
            self.stats.update_transition2(
                self.opp_hist[-2], self.opp_hist[-1], opp_action
            )
        if self.my_hist and self.opp_hist:
            self.stats.update_battle(my_action, opp_action)
        
        # 更新历史
        self.opp_hist.append(opp_action)
        self.my_hist.append(my_action)
        self.rounds_played += 1
        
        # 尝试训练
        self._maybe_fit()
    
    def batch_play(self, my_actions, opp_actions):
        """批量更新"""
        for a_my, a_opp in zip(my_actions, opp_actions):
            self.play(int(a_my), int(a_opp))
    
    def getscores(self):
        return self.score
    
    def save(self, idx: Optional[int] = None) -> None:
        """保存模型"""
        import os, pickle
        os.makedirs("models", exist_ok=True)
        
        if _XGB_OK and self.model is not None:
            try:
                # 保存 XGBoost 模型
                path = f"models/{self.idxname}_agent.json"
                self.model.save_model(path)
                
                # 保存状态
                state_path = f"models/{self.idxname}_state.pkl"
                with open(state_path, "wb") as f:
                    pickle.dump({
                        "ctx_len": self.ctx_len,
                        "opp_hist": self.opp_hist[-1000:],
                        "my_hist": self.my_hist[-1000:],
                        "rounds_played": self.rounds_played,
                        "stats": self.stats
                    }, f)
                return
            except Exception:
                pass
        
        # 备用保存
        with open(f"models/{self.idxname}_agent.pkl", "wb") as f:
            pickle.dump({
                "X": list(self.X)[-1000:],
                "y": list(self.y)[-1000:],
                "ctx_len": self.ctx_len,
                "opp_hist": self.opp_hist[-1000:],
                "my_hist": self.my_hist[-1000:],
                "stats": self.stats
            }, f)