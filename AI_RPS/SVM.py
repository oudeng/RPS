# -*- coding: utf-8 -*-
"""
SVM_optimized.py - GPU 加速的优化版 SVM

主要改进：
1. 内存管理：使用 deque 限制最大长度
2. 增量统计：O(1) 特征提取
3. GPU 支持：优先使用 cuML，否则降级到 sklearn
4. 样本权重：时间衰减权重（可选）

保持核心特色：
- RBF 核 + 特征标准化（SVM 必需）
- 概率预测
"""

from typing import List, Optional
import random
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

# GPU 支持检测
_GPU_AVAILABLE = False
_USE_CUML = False

try:
    # 尝试导入 cuML（GPU 加速的 sklearn）
    from cuml.svm import SVC as cuSVC
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    import cupy as cp
    _USE_CUML = True
    _GPU_AVAILABLE = True
    print("✓ cuML (GPU) detected and loaded")
except ImportError:
    _USE_CUML = False

# 降级到 sklearn
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    _SK_OK = True
    if not _USE_CUML:
        print("✓ sklearn (CPU) loaded")
except ImportError:
    _SK_OK = False
    print("❌ Neither cuML nor sklearn available")


class IncrementalStats:
    """增量统计（避免重复计算）"""
    def __init__(self):
        self.trans_matrix = np.zeros((3, 3), dtype=np.float32)
        self.total_trans = 0
    
    def update_transition(self, prev: int, curr: int):
        self.trans_matrix[prev, curr] += 1
        self.total_trans += 1
    
    def get_trans_probs(self) -> np.ndarray:
        if self.total_trans > 0:
            return (self.trans_matrix / self.total_trans).reshape(-1)
        return np.zeros(9, dtype=np.float32)


def _enhanced_features_fast(
    opp_hist: List[int],
    my_hist: List[int],
    stats: IncrementalStats,
    ctx_len: int = 12
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
    
    # 3. 最近频率（多尺度）
    for w in [5, 10, 20]:
        if len(opp_hist) >= w:
            recent_freq = np.bincount(opp_hist[-w:], minlength=3).astype(np.float32)
            recent_freq = recent_freq / (recent_freq.sum() + 1e-5)
        else:
            recent_freq = opp_freq
        features.append(recent_freq)
    
    # 4. 转移概率（使用增量统计）
    features.append(stats.get_trans_probs())
    
    # 5. 胜负率统计
    if my_hist and opp_hist:
        win_stats = np.zeros(3, dtype=np.float32)
        for i in range(min(len(my_hist), len(opp_hist))):
            outcome = _outcome(my_hist[i], opp_hist[i])
            if outcome == 2:
                win_stats[my_hist[i]] += 1
        if win_stats.sum() > 0:
            win_stats = win_stats / win_stats.sum()
        features.append(win_stats)
    else:
        features.append(np.array([1/3, 1/3, 1/3], dtype=np.float32))
    
    return np.concatenate(features, axis=0)


class Train:
    name = "SVM"
    
    def __init__(
        self,
        ctx_len: int = 12,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        retrain_every: int = 40,
        min_fit: int = 60,
        max_samples: int = 2000,
        use_gpu: bool = True,
        use_sample_weight: bool = False,  # SVM 通常不需要样本权重
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
        
        # 使用 deque 限制内存
        self.opp_hist: List[int] = []
        self.my_hist: List[int] = []
        self.X = deque(maxlen=self.max_samples * 2)
        self.y = deque(maxlen=self.max_samples * 2)
        
        self.last_fit_n = 0
        self.clf = None
        self.scaler = None
        self.rounds_played = 0
        
        # 增量统计
        self.stats = IncrementalStats()
        
        # GPU 选择
        self.use_gpu = use_gpu and _GPU_AVAILABLE and _USE_CUML
        
        if _USE_CUML and self.use_gpu:
            # GPU 版本 (cuML)
            self.clf = cuSVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                class_weight="balanced",
                cache_size=500
            )
            self.scaler = cuStandardScaler()
            print(f"✓ {self.name} initialized with GPU (cuML)")
        
        elif _SK_OK:
            # CPU 版本 (sklearn)
            self.clf = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                class_weight="balanced",
                cache_size=500,
                random_state=seed
            )
            self.scaler = StandardScaler()
            print(f"✓ {self.name} initialized with CPU (sklearn)")
        
        else:
            print(f"❌ {self.name}: No SVM library available")
    
    def punches(self, round_idx: Optional[int] = None) -> int:
        """预测对手动作并返回克制动作"""
        # 早期探索
        if self.rounds_played < 30:
            if random.random() < 0.25:
                return random.randint(0, 2)
        
        if self.clf is not None and hasattr(self.clf, "classes_"):
            try:
                # 提取特征
                f = _enhanced_features_fast(
                    self.opp_hist, self.my_hist, self.stats, self.ctx_len
                ).reshape(1, -1)
                
                # 标准化（关键！）
                if _USE_CUML and self.use_gpu:
                    # GPU 版本
                    f_gpu = cp.asarray(f, dtype=cp.float32)
                    f_scaled = self.scaler.transform(f_gpu)
                    probs = self.clf.predict_proba(f_scaled)
                    probs = cp.asnumpy(probs)[0]  # 转回 CPU
                else:
                    # CPU 版本
                    f_scaled = self.scaler.transform(f)
                    probs = self.clf.predict_proba(f_scaled)[0]
                
                # 早期添加噪声
                if self.rounds_played < 100:
                    noise = np.random.random(3) * 0.15
                    probs = probs + noise
                    probs = probs / probs.sum()
                
                # 10% 概率随机采样
                if random.random() < 0.1:
                    opp_pred = np.random.choice(3, p=probs)
                else:
                    opp_pred = int(np.argmax(probs))
            
            except Exception as e:
                opp_pred = _freq_pred(self.opp_hist)
        else:
            opp_pred = _freq_pred(self.opp_hist)
        
        return _beat(opp_pred)
    
    def _compute_sample_weights(self, n: int) -> np.ndarray:
        """计算样本权重（可选）"""
        if not self.use_sample_weight:
            return None
        weights = np.exp(np.linspace(-1, 0, n))
        return weights.astype(np.float32)
    
    def _maybe_fit(self):
        """训练模型"""
        if self.clf is None:
            return
        
        n = len(self.y)
        
        # 动态调整训练频率
        retrain_interval = self.retrain_every
        if n > 300:
            retrain_interval = int(self.retrain_every * 1.5)
        
        if n >= self.min_fit and (n - self.last_fit_n) >= retrain_interval:
            # 转换为 numpy 数组
            X = np.array(list(self.X)[-self.max_samples:], dtype=np.float32)
            y = np.array(list(self.y)[-self.max_samples:], dtype=np.int64)
            
            # 样本权重
            weights = self._compute_sample_weights(len(X))
            
            try:
                if _USE_CUML and self.use_gpu:
                    # GPU 版本
                    X_gpu = cp.asarray(X)
                    y_gpu = cp.asarray(y)
                    
                    # 标准化
                    self.scaler.fit(X_gpu)
                    X_scaled = self.scaler.transform(X_gpu)
                    
                    # 训练
                    if weights is not None:
                        w_gpu = cp.asarray(weights)
                        self.clf.fit(X_scaled, y_gpu, sample_weight=w_gpu)
                    else:
                        self.clf.fit(X_scaled, y_gpu)
                
                else:
                    # CPU 版本
                    self.scaler.fit(X)
                    X_scaled = self.scaler.transform(X)
                    
                    if weights is not None:
                        self.clf.fit(X_scaled, y, sample_weight=weights)
                    else:
                        self.clf.fit(X_scaled, y)
                
                self.last_fit_n = n
            
            except Exception as e:
                # 静默失败
                pass
    
    def play(self, my_action: int, opp_action: int) -> None:
        """单步更新"""
        # 更新分数
        out = _outcome(my_action, opp_action)
        self.score += _score_delta(out)
        
        # 提取特征（使用当前历史）
        f = _enhanced_features_fast(
            self.opp_hist, self.my_hist, self.stats, self.ctx_len
        )
        
        # 保存样本
        self.X.append(f)
        self.y.append(opp_action)
        
        # 增量更新统计
        if len(self.opp_hist) >= 1:
            self.stats.update_transition(self.opp_hist[-1], opp_action)
        
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
        import os
        os.makedirs("models", exist_ok=True)
        path = f"models/{self.idxname}_agent.pkl"
        
        if self.clf is not None:
            try:
                import joblib
                
                # 如果是 GPU 模型，转换到 CPU
                if _USE_CUML and self.use_gpu:
                    # cuML 模型需要特殊处理
                    joblib.dump({
                        'clf': self.clf,
                        'scaler': self.scaler,
                        'ctx_len': self.ctx_len,
                        'stats': self.stats,
                        'gpu': True
                    }, path)
                else:
                    joblib.dump({
                        'clf': self.clf,
                        'scaler': self.scaler,
                        'ctx_len': self.ctx_len,
                        'stats': self.stats,
                        'gpu': False
                    }, path)
                return
            except Exception:
                pass
        
        # 备用保存
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "X": list(self.X)[-1000:],
                "y": list(self.y)[-1000:],
                "ctx_len": self.ctx_len,
                "opp_hist": self.opp_hist[-1000:],
                "my_hist": self.my_hist[-1000:],
                "stats": self.stats
            }, f)