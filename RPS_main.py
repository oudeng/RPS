#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_v4_multi_seed.py — RPS v4.0 完整测试系统

on Jan 10, 2026

主要功能：
1. 支持混合已训练和未训练的agent
2. Lipschitz界分析（改进版）
3. 保存模型到models_seed*目录
4. 生成experiment_metadata.json
5. 生成完整的分析所需文件
6. 兼容所有分析脚本

Usage example (as the seat of "test_3_1_RNNvsA3C.csv"):
1. 基本测试运行
python test_v4_multi_seed.py \
    --rounds 500 \
    --seeds 1,2,3,5,8 \
    --seats Agent_seats/test_3_1_RNNvsA3C.csv \
    --input-dir RPS_train_summary \
    --output-dir Test_Results_v4

2. 带Lipschitz分析的运行(pairwise only!!)
python test_v4_multi_seed.py \
    --rounds 500 \
    --seeds 1,2,3 \
    --seats Agent_seats/test_3_1_RNNvsA3C.csv \
    --input-dir RPS_train_summary \
    --output-dir Test_Lipschitz \
    --history-k 20 \
    --warmup 50 \
    --debug

3. 使用one-hot分布
python test_v4_multi_seed.py \
    --rounds 500 \
    --seeds 1,2,3 \
    --seats Agent_seats/test_3_1_RNNvsA3C.csv \
    --input-dir RPS_train_summary \
    --output-dir Test_OneHot \
    --use-onehot \
    --warmup 50

# 参数说明
## 基本参数
--rounds: 每对agent对战轮数（默认500）
--seeds: 测试种子列表，逗号分隔（如"1,2,3,5,8"）
--seats: 座位配置文件路径
--input-dir: 已训练模型目录（默认"RPS_train_summary"）
--output-dir: 输出目录（必需）

## 游戏参数
--games-per-pair: 每轮对战次数（默认1）
--batch-size: 批处理大小（默认32）
--batch-freq: 批更新频率（默认32）

## Lipschitz参数
--history-k: 经验分布的历史窗口（默认20）
--use-onehot: 使用one-hot分布而非窗口分布
--warmup: 预热轮数，跳过初始不稳定期（默认50）

## 调试参数
--debug: 启用调试输出
"""

import os
import argparse
import random
import json
import re
import pickle
import glob
import shutil
import importlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==================
# GPU优化
# ==================
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ==================
# Agent配置
# ==================
AGENT_MODULES = {
    "LSTM_v1": "AI_RPS.LSTM_v1",
    "LSTM_v2": "AI_RPS.LSTM_v2",
    "A3C_v1": "AI_RPS.A3C_v1",
    "A3C_v2": "AI_RPS.A3C_v2",
    "MSA_v1": "AI_RPS.MSA_v1",
    "MSA_v2": "AI_RPS.MSA_v2",
    "Tr_v1": "AI_RPS.Tr_v1",
    "Tr_v2": "AI_RPS.Tr_v2",
    "RNN_v1": "AI_RPS.RNN_v1",
    "RNN_v2": "AI_RPS.RNN_v2",
    "B_v1": "AI_RPS.B_v1",
    "B_v2": "AI_RPS.B_v2",
    "M_v1": "AI_RPS.M_v1",
    "M_v2": "AI_RPS.M_v2",
    "R": "AI_RPS.R",
    "WL": "AI_RPS.WL",
    "CG": "AI_RPS.CG",
    "RF": "AI_RPS.RF",
    "SVM": "AI_RPS.SVM",
    "XGB": "AI_RPS.XGB",
    "A3C_v2un": "AI_RPS.A3C_v2un",
    "LSTM_v2un": "AI_RPS.LSTM_v2un",
    "MSA_v2un": "AI_RPS.MSA_v2un",
    "RNN_v2un": "AI_RPS.RNN_v2un",
    "Tr_v2un": "AI_RPS.Tr_v2un",
}

# 模型分类
NEURAL_MODELS = {"LSTM_v1","LSTM_v2","A3C_v1","A3C_v2","MSA_v1","MSA_v2","Tr_v1","Tr_v2","RNN_v1","RNN_v2","A3C_v2un","LSTM_v2un","MSA_v2un","RNN_v2un","Tr_v2un"}
STATISTICAL_MODELS = {"B_v1", "B_v2", "M_v1", "M_v2"}
ML_MODELS = {"RF", "SVM", "XGB"}
SIMPLE_MODELS = {"R", "WL", "CG"}
BATCH_CAPABLE_MODELS = {"LSTM_v1","LSTM_v2","A3C_v1","A3C_v2","MSA_v1","MSA_v2","Tr_v1","Tr_v2","RNN_v1","RNN_v2","A3C_v2un","LSTM_v2un","MSA_v2un","RNN_v2un","Tr_v2un"}

ACTIONS = ["Rock", "Paper", "Scissors"]


# ==================
# 座位配置
# ==================
@dataclass
class TestSeatCfg:
    seat: int                    # 测试席位号
    agent_full: str             # 完整名称（如 "51_RNN"）
    method: str                 # 方法名（如 "RNN"）
    train_seed: Optional[int]   # 训练种子（None表示未训练）
    init_kwargs: Dict[str, Any] = field(default_factory=dict)  # 传给agent构造函数的额外超参（从hp_*列解析）


# ==================
# 工具函数
# ==================
def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_to_name(a: int) -> str:
    try:
        return ACTIONS[a]
    except Exception:
        return str(a)


def judge(who_a: int, whom_a: int) -> Tuple[int, str]:
    """返回 (score_delta_who, winner_tag)"""
    out = (whom_a - who_a) % 3
    if out == 2:
        return +1, "who"
    elif out == 1:
        return -1, "whom"
    else:
        return 0, "tie"


def _parse_method_from_agent(agent_name: str) -> str:
    """从 '51_RNN' 中提取 'RNN'"""
    s = str(agent_name).strip()
    
    # 首先检查是否是纯方法名
    if s in AGENT_MODULES:
        return s
    
    # 尝试从 "数字_方法名" 格式中提取
    if "_" in s:
        parts = s.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[1]
    
    return s


def load_test_seats(path: str) -> List[TestSeatCfg]:
    """加载座位配置文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Seats file not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    
    # 必须有seat和agent列
    for col in ["seat", "agent"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    
    seats: List[TestSeatCfg] = []
    for _, row in df.iterrows():
        seat_id = int(row["seat"])
        agent_spec = str(row["agent"]).strip()
        idxname = None
        if "idxname" in df.columns and pd.notna(row.get("idxname", None)):
            idxname = str(row["idxname"]).strip()
        agent_full = idxname if idxname else agent_spec
        
        # 检查seed列
        train_seed = None
        if "seed" in df.columns:
            seed_value = row["seed"]
            if pd.notna(seed_value) and str(seed_value).strip() != "":
                try:
                    train_seed = int(seed_value)
                except (ValueError, TypeError):
                    train_seed = None
        

        # 解析超参：仅接受以 hp_ 开头的列，避免把分析列（score/wins等）误当作超参
        init_kwargs: Dict[str, Any] = {}
        for col in df.columns:
            if not str(col).startswith("hp_"):
                continue
            key = str(col)[3:]
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            # 类型规整：把 32.0 -> 32，把 "1e-3" -> 0.001 等
            if isinstance(val, (int, float, np.integer, np.floating)):
                if float(val).is_integer():
                    val = int(val)
                else:
                    val = float(val)
            else:
                s = str(val).strip()
                try:
                    if re.match(r"^[+-]?\d+$", s):
                        val = int(s)
                    else:
                        val = float(s)
                except Exception:
                    val = s
            init_kwargs[key] = val

        # 解析方法名
        method = _parse_method_from_agent(agent_spec)
        if method not in AGENT_MODULES:
            raise ValueError(f"Unknown method '{method}' from agent '{agent_full}'")
        
        seats.append(TestSeatCfg(
            seat=seat_id,
            agent_full=agent_full,
            method=method,
            train_seed=train_seed,
            init_kwargs=init_kwargs
        ))
    
    seats.sort(key=lambda x: x.seat)
    return seats


def _import_agent(method_name: str):
    """导入agent类"""
    if method_name not in AGENT_MODULES:
        raise ValueError(f"Unknown method: {method_name}")
    module_path = AGENT_MODULES[method_name]
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "Train"):
        raise AttributeError(f"Module {module_path} has no Train class")
    return mod.Train


def _create_agent_with_idxname(cls, idxname: str, init_kwargs: Optional[Dict[str, Any]] = None):
    """创建agent实例并设置idxname（改进版）"""
    import traceback
    
    agent = None
    errors = []
    
    # 尝试不同的创建方式
    init_kwargs = init_kwargs or {}
    # 尝试不同的创建方式（带超参 / 不带超参）
    candidate_kwargs = [
        {**init_kwargs, "idxname": idxname},
        {**init_kwargs, "name": idxname},
        {**init_kwargs},
        {"idxname": idxname},
        {"name": idxname},
        {}
    ]
    for kwargs in candidate_kwargs:
        try:
            agent = cls(**kwargs)
            break
        except Exception as e:  # ← 捕获所有异常
            error_msg = f"kwargs={kwargs}: {type(e).__name__}: {e}"
            errors.append(error_msg)
            
            # 打印详细错误
            if not isinstance(e, TypeError):
                print(f"\n❌ Agent creation failed:")
                print(f"   Class: {cls.__name__}")
                print(f"   Kwargs: {kwargs}")
                print(f"   Error: {e}")
                traceback.print_exc()
            continue
    
    if agent is None:
        print(f"\n❌ Failed to create agent of class {cls.__name__}")
        print("All attempts failed:")
        for err in errors:
            print(f"  - {err}")
        raise RuntimeError(f"Failed to create agent of class {cls.__name__}")
    
    # 确保设置idxname
    if not hasattr(agent, "idxname") or agent.idxname != idxname:
        agent.idxname = idxname
    
    return agent


# ==================
# 统一模型保存/加载系统
# ==================
def _save_agent_unified(agent, filepath: str, metadata: dict = None) -> bool:
    """
    统一的模型保存函数
    
    保存格式：
    {
        'save_version': 'unified_v1',
        'agent_type': agent类名,
        'agent_name': agent.idxname,
        'model_state_dict': {...},      # 主模型参数
        'optimizer_state_dict': {...},   # 优化器参数（如果有）
        'extra_state': {...},           # 其他状态（如统计信息）
        'metadata': {...}               # 元信息
    }
    
    Returns:
        bool: 保存成功返回True，否则返回False
    """
    try:
        save_dict = {
            'save_version': 'unified_v1',
            'agent_type': type(agent).__name__,
            'agent_name': getattr(agent, 'idxname', 'unknown'),
        }
        
        # 保存模型参数 - 尝试多个可能的属性
        model_saved = False
        for model_attr in ['model', 'net', 'policy_net', 'actor', 'critic', 'value_net']:
            if hasattr(agent, model_attr):
                model = getattr(agent, model_attr)
                if hasattr(model, 'state_dict'):
                    save_dict['model_state_dict'] = model.state_dict()
                    model_saved = True
                    break
        
        if not model_saved:
            # 没有找到模型参数，可能是统计模型
            pass
        
        # 保存优化器状态
        if hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'state_dict'):
            try:
                save_dict['optimizer_state_dict'] = agent.optimizer.state_dict()
            except Exception:
                pass
        
        # 保存额外状态（统计模型的计数等）
        extra_state = {}
        for attr in ['counts', 'trans', 'history', 'opp_hist', 'last_opp', 
                     'last_action', 'last_policy', 'episode', 'step']:
            if hasattr(agent, attr):
                val = getattr(agent, attr)
                # 只保存可序列化的类型
                try:
                    if isinstance(val, (int, float, str, bool)):
                        extra_state[attr] = val
                    elif isinstance(val, (list, tuple)):
                        extra_state[attr] = val
                    elif isinstance(val, dict):
                        extra_state[attr] = val
                    elif isinstance(val, np.ndarray):
                        extra_state[attr] = val.tolist()
                    elif isinstance(val, deque):
                        extra_state[attr] = {
                            'type': 'deque',
                            'data': list(val),
                            'maxlen': val.maxlen
                        }
                except Exception:
                    pass  # 跳过无法序列化的属性
        
        if extra_state:
            save_dict['extra_state'] = extra_state
        
        # 保存元信息
        if metadata:
            save_dict['metadata'] = metadata
        
        # 保存到文件
        torch.save(save_dict, filepath)
        return True
        
    except Exception as e:
        print(f"  [warn] Unified save failed for {filepath}: {e}")
        return False


def _load_agent_unified(agent, filepath: str, debug: bool = False) -> bool:
    """
    统一的模型加载函数
    
    尝试加载统一格式的模型文件。如果不是统一格式或加载失败，返回False。
    
    Returns:
        bool: 加载成功返回True，否则返回False
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # 检查是否是统一格式
        if not isinstance(checkpoint, dict) or checkpoint.get('save_version') != 'unified_v1':
            if debug:
                print(f"  [debug] Not unified format")
            return False
        
        if debug:
            print(f"  [debug] Loading unified format v1")
            print(f"  [debug] Agent type: {checkpoint.get('agent_type')}")
            print(f"  [debug] Agent name: {checkpoint.get('agent_name')}")
        
        # 加载模型参数
        if 'model_state_dict' in checkpoint:
            model_loaded = False
            for model_attr in ['model', 'net', 'policy_net', 'actor', 'critic', 'value_net']:
                if hasattr(agent, model_attr):
                    model = getattr(agent, model_attr)
                    if hasattr(model, 'load_state_dict'):
                        try:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            model_loaded = True
                            if debug:
                                print(f"  [debug] Loaded model to {model_attr}")
                            break
                        except Exception as e:
                            if debug:
                                print(f"  [debug] Failed to load to {model_attr}: {e}")
                            continue
            
            if not model_loaded and debug:
                print(f"  [debug] Could not find suitable model attribute")
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint and hasattr(agent, 'optimizer'):
            try:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if debug:
                    print(f"  [debug] Loaded optimizer state")
            except Exception as e:
                if debug:
                    print(f"  [debug] Could not load optimizer: {e}")
        
        # 加载额外状态
        if 'extra_state' in checkpoint:
            for key, value in checkpoint['extra_state'].items():
                try:
                    # 恢复原始类型
                    if isinstance(value, dict) and value.get('type') == 'deque':
                        # 恢复deque
                        restored = deque(value['data'], maxlen=value.get('maxlen'))
                        setattr(agent, key, restored)
                    elif isinstance(value, list) and hasattr(agent, key):
                        original = getattr(agent, key)
                        if isinstance(original, np.ndarray):
                            setattr(agent, key, np.array(value))
                        else:
                            setattr(agent, key, value)
                    else:
                        setattr(agent, key, value)
                except Exception as e:
                    if debug:
                        print(f"  [debug] Could not restore {key}: {e}")
        
        print(f"[load] {filepath} (unified format v1)")
        return True
        
    except Exception as e:
        if debug:
            print(f"  [debug] Unified load error: {e}")
        return False


def _load_pretrained_for_agent(agent, idxname_train: str, train_seed: int, input_dir: str, debug: bool = False):
    """加载预训练模型 - 优先使用统一格式，然后回退到兼容模式"""
    model_dir = os.path.join(input_dir, f"models_seed{train_seed}")
    if not os.path.exists(model_dir):
        print(f"[warn] model_dir not found for seed {train_seed}: {model_dir}")
        return
    
    base = os.path.join(model_dir, f"{idxname_train}_agent")
    candidates = [base + ".pth", base + ".pkl", base + ".json"]
    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break
    
    if not found:
        print(f"[info] no pretrained file for {idxname_train} in {model_dir}")
        return
    
    try:
        # 方法0: 优先尝试统一格式（推荐格式）
        if found.endswith('.pth'):
            if _load_agent_unified(agent, found, debug):
                return  # 成功加载统一格式，直接返回
        
        # 如果不是统一格式或加载失败，继续尝试兼容模式
        
        # 方法1: 尝试使用 load_state_dict (PyTorch标准方式) - agent级别
        if hasattr(agent, 'load_state_dict') and found.endswith('.pth'):
            state_dict = torch.load(found, map_location='cpu', weights_only=False)
            agent.load_state_dict(state_dict, strict=False)
            print(f"[load] {found} (via agent.load_state_dict)")
            return
        
        # 方法2: 尝试使用 load() 方法
        if hasattr(agent, 'load'):
            import inspect
            sig = inspect.signature(agent.load)
            # 检查 load 方法的参数数量（排除 self）
            num_params = len([p for p in sig.parameters.values() 
                            if p.default == inspect.Parameter.empty and p.name != 'self'])
            
            if num_params == 0:
                # load() 不接受参数，需要先设置路径
                if hasattr(agent, 'model_path'):
                    agent.model_path = found
                agent.load()
                print(f"[load] {found} (via load() no-arg)")
                return
            else:
                # load() 接受文件路径参数
                agent.load(found)
                print(f"[load] {found} (via load(path))")
                return
        
        # 方法3: 尝试直接加载整个agent对象（pickle）
        if found.endswith('.pkl'):
            with open(found, 'rb') as f:
                loaded_agent = pickle.load(f)
            # 复制关键属性
            for attr in ['model', 'net', 'policy_net', 'value_net', 
                        'optimizer', 'counts', 'trans', 'history']:
                if hasattr(loaded_agent, attr):
                    setattr(agent, attr, getattr(loaded_agent, attr))
            print(f"[load] {found} (via pickle)")
            return
        
        # 方法4: 尝试从.pth文件加载到模型
        if found.endswith('.pth'):
            checkpoint = torch.load(found, map_location='cpu', weights_only=False)
            
            if debug:
                print(f"  [debug] checkpoint type: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    print(f"  [debug] checkpoint keys: {checkpoint.keys()}")
            
            # 如果是字典，尝试提取模型参数
            if isinstance(checkpoint, dict):
                loaded = False
                
                # 尝试常见的键名
                for key in ['model_state_dict', 'state_dict', 'model', 'net', 'policy_net']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        
                        # 尝试加载到不同的模型属性
                        for model_attr in ['model', 'net', 'policy_net']:
                            if hasattr(agent, model_attr):
                                model = getattr(agent, model_attr)
                                try:
                                    model.load_state_dict(state_dict, strict=False)
                                    print(f"[load] {found} (via checkpoint['{key}'] -> {model_attr}, strict=False)")
                                    loaded = True
                                    break
                                except Exception as e:
                                    if debug:
                                        print(f"  [debug] Failed to load to {model_attr}: {e}")
                                    continue
                        
                        if loaded:
                            # 如果有optimizer等其他信息也尝试加载
                            if 'optimizer_state_dict' in checkpoint and hasattr(agent, 'optimizer'):
                                try:
                                    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                except Exception:
                                    pass
                            return
                
                # 如果上面都失败了，尝试直接将整个checkpoint当作state_dict
                if not loaded:
                    for model_attr in ['model', 'net', 'policy_net']:
                        if hasattr(agent, model_attr):
                            model = getattr(agent, model_attr)
                            try:
                                model.load_state_dict(checkpoint, strict=False)
                                print(f"[load] {found} (via direct checkpoint -> {model_attr}, strict=False)")
                                return
                            except Exception as e:
                                if debug:
                                    print(f"  [debug] Failed direct load to {model_attr}: {e}")
                                continue
            else:
                # checkpoint 本身就是 state_dict
                for model_attr in ['model', 'net', 'policy_net']:
                    if hasattr(agent, model_attr):
                        model = getattr(agent, model_attr)
                        try:
                            model.load_state_dict(checkpoint, strict=False)
                            print(f"[load] {found} (via direct state_dict -> {model_attr}, strict=False)")
                            return
                        except Exception as e:
                            if debug:
                                print(f"  [debug] Failed to load to {model_attr}: {e}")
                            continue
        
        # 如果所有方法都失败了，打印调试信息
        print(f"[warn] No suitable loading method found for {found}")
        if debug:
            print(f"  [debug] Agent type: {type(agent).__name__}")
            print(f"  [debug] Agent attributes: {[a for a in dir(agent) if not a.startswith('_')]}")
        
    except Exception as e:
        print(f"[warn] Failed to load {found}: {e}")
        if debug:
            import traceback
            traceback.print_exc()


# ==================
# Lipschitz分析功能
# ==================
def compute_empirical_distribution(history: deque, k: int = 20) -> np.ndarray:
    """计算最近k步的经验分布"""
    if len(history) == 0:
        return np.array([1/3, 1/3, 1/3])
    
    recent = list(history)[-k:] if len(history) >= k else list(history)
    if len(recent) == 0:
        return np.array([1/3, 1/3, 1/3])
    
    counts = np.zeros(3)
    for action in recent:
        if 0 <= action <= 2:
            counts[action] += 1
    
    if counts.sum() > 0:
        return counts / counts.sum()
    else:
        return np.array([1/3, 1/3, 1/3])


def compute_onehot_distribution(last_action: int) -> np.ndarray:
    """使用单步分布（one-hot）"""
    p = np.zeros(3)
    if 0 <= last_action <= 2:
        p[last_action] = 1.0
    else:
        p[:] = 1/3
    return p


def extract_prediction_distribution(agent, round_num: int, debug: bool = False, 
                                   allow_hist_fallback: bool = False) -> Tuple[np.ndarray, str]:
    """从agent中提取预测分布"""
    default_dist = np.array([1/3, 1/3, 1/3])
    agent_name = getattr(agent, 'name', getattr(agent, '__class__.__name__', 'Unknown'))
    
    try:
        # Prefer last_policy / policy vector if available.
        # Robust to slightly unnormalized probabilities and logits-like outputs.
        if hasattr(agent, 'last_policy') and agent.last_policy is not None:
            raw = np.asarray(agent.last_policy, dtype=float).reshape(-1)
            dist = None
            if raw.size == 3 and np.all(np.isfinite(raw)):
                # Case 1: already looks like probabilities (allow tiny negatives due to numeric noise)
                if np.all(raw >= -1e-6):
                    raw = np.clip(raw, 0.0, None)
                    s = raw.sum()
                    if s > 0:
                        dist = raw / s

                # Case 2: treat as logits -> softmax
                if dist is None:
                    z = raw - np.max(raw)
                    expz = np.exp(z)
                    s = expz.sum()
                    if s > 0:
                        dist = expz / s

            if dist is not None and dist.shape == (3,):
                if debug:
                    print(f"  [{agent_name}] Using last_policy: {dist}")
                return dist.astype(float), "policy"
        
        # 统计模型
        if agent_name in STATISTICAL_MODELS or agent_name in ["B_v1", "B_v2"]:
            if hasattr(agent, 'counts'):
                counts = np.asarray(agent.counts, dtype=float)
                if counts.size == 3 and counts.sum() > 0:
                    dist = counts / counts.sum()
                    if debug:
                        print(f"  [{agent_name}] Counts-based dist: {dist}")
                    return dist, "stat_counts"
        
        # 马尔可夫模型
        if agent_name in ["M_v1", "M_v2"]:
            if hasattr(agent, 'trans') and hasattr(agent, 'last_opp'):
                if agent.last_opp is not None:
                    trans_row = np.asarray(agent.trans[agent.last_opp], dtype=float)
                    if trans_row.size == 3 and trans_row.sum() > 0:
                        dist = trans_row / trans_row.sum()
                        if debug:
                            print(f"  [{agent_name}] Markov dist: {dist}")
                        return dist, "markov"
        
        # ML模型
        if agent_name in ML_MODELS:
            if hasattr(agent, 'model') and hasattr(agent.model, 'predict_proba'):
                if hasattr(agent, 'X_hist') and len(agent.X_hist) > 0:
                    try:
                        probs = agent.model.predict_proba(agent.X_hist[-1].reshape(1, -1))[0]
                        if len(probs) == 3:
                            dist = np.array(probs, dtype=float)
                            dist = dist / dist.sum()
                            if debug:
                                print(f"  [{agent_name}] ML proba: {dist}")
                            return dist, "ml_proba"
                    except Exception:
                        pass
        
        # 历史频率（仅当允许时）
        if allow_hist_fallback and hasattr(agent, 'opp_hist'):
            if len(agent.opp_hist) > 0:
                hist = list(agent.opp_hist)[-20:]
                counts = np.zeros(3)
                for a in hist:
                    if 0 <= a <= 2:
                        counts[a] += 1
                if counts.sum() > 0:
                    dist = counts / counts.sum()
                    if debug:
                        print(f"  [{agent_name}] Hist fallback: {dist}")
                    return dist, "hist_fallback"
        
        # 简单策略
        if agent_name == "R":
            return default_dist, "random"
        elif agent_name == "CG":
            return np.array([1.0, 0.0, 0.0]), "constant"
        elif agent_name == "WL":
            return default_dist, "reactive"
        
    except Exception as e:
        if debug:
            print(f"  [{agent_name}] Error: {e}")
    
    return default_dist, "uniform"


def compute_l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute L1 distance |p-q|_1."""
    return float(np.abs(p - q).sum())


# Payoff matrix (row = our action, col = opponent action)
# Standard RPS: win=+1, lose=-1, tie=0  (payoff range [-1, 1])
PAYOFF_MATRIX = np.array([
    [0, -1, 1],   # Rock vs (Rock, Paper, Scissors)
    [1, 0, -1],   # Paper vs (Rock, Paper, Scissors)
    [-1, 1, 0],   # Scissors vs (Rock, Paper, Scissors)
], dtype=float)


def best_response_action(p_dist: np.ndarray) -> int:
    """Best-response action against opponent distribution p_dist."""
    expected = PAYOFF_MATRIX @ p_dist
    return int(np.argmax(expected))


def compute_expected_payoff(action: int, p_dist: np.ndarray) -> float:
    """Expected payoff of `action` against opponent distribution p_dist."""
    return float(PAYOFF_MATRIX[action] @ p_dist)


def compute_regret(action: int, p_true: np.ndarray) -> float:
    """Regret: Δ = U*(p_true) - U(action, p_true), with payoff in {-1,0,+1}."""
    u_actual = compute_expected_payoff(action, p_true)
    u_optimal = float(np.max(PAYOFF_MATRIX @ p_true))
    return u_optimal - u_actual


# ==================
# 主测试系统
# ==================
class TestSystemV4:
    """V4版本测试系统：完整输出+Lipschitz分析"""
    
    def __init__(self, test_seats: List[TestSeatCfg], input_dir: str, output_dir: str,
                 seed: int, games_per_pair: int = 1, batch_size: int = 32,
                 batch_freq: int = 32, history_k: int = 20, use_onehot: bool = False,
                 warmup_rounds: int = 50, debug: bool = False, **kwargs):
        
        self.test_seats = test_seats
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.seed = seed
        self.games_per_pair = games_per_pair
        self.batch_size = batch_size
        self.batch_freq = batch_freq
        self.history_k = history_k
        self.use_onehot = use_onehot
        self.warmup_rounds = warmup_rounds
        self.debug = debug
        # 保存额外参数但不使用（为了兼容性）
        self.extra_kwargs = kwargs
        
        # 设置种子
        _set_global_seed(seed)
        
        # 初始化agents
        self.agents = self._init_agents()
        
        # 历史记录
        self.opponent_histories = defaultdict(lambda: deque(maxlen=100))
        self.lipschitz_data = []
        self.stats = {
            'l1_distances': [],
            'regrets': [],
            'pred_sources': []
        }
    
    def _init_agents(self):
        """初始化所有agents"""
        agents = []
        
        for cfg in self.test_seats:
            cls = _import_agent(cfg.method)
            # Pass init_kwargs (parsed from hp_* columns) to agent constructor
            agent = _create_agent_with_idxname(cls, cfg.agent_full, cfg.init_kwargs)
            
            # 设置batch size（如果支持）
            if cfg.method in BATCH_CAPABLE_MODELS:
                try:
                    agent.set_batch_size(self.batch_size)
                except Exception:
                    pass
            
            # 加载预训练模型（如果有）
            if cfg.train_seed is not None:
                _load_pretrained_for_agent(
                    agent, cfg.agent_full, cfg.train_seed, self.input_dir, self.debug
                )
            else:
                print(f"[init] {cfg.agent_full}: starting from fresh state (untrained)")
            
            agents.append(agent)
        
        return agents
    
    def _save_models(self):
        """保存所有模型到models_seed*目录 - 使用统一格式"""
        model_dir = os.path.join(self.output_dir, f"models_seed{self.seed}")
        os.makedirs(model_dir, exist_ok=True)
        print(f"  [models] saving to {model_dir}")
        
        # 直接保存到目标目录，使用统一格式
        for agent in self.agents:
            idx = getattr(agent, "idxname", None)
            if not idx:
                continue
            
            # 准备元信息
            metadata = {
                'seed': self.seed,
                'timestamp': datetime.now().isoformat(),
                'batch_size': self.batch_size,
            }
            
            # 统一格式文件路径
            unified_path = os.path.join(model_dir, f"{idx}_agent.pth")
            
            # 尝试统一格式保存
            if _save_agent_unified(agent, unified_path, metadata):
                print(f"  [saved] {idx} → {unified_path} (unified format)")
            else:
                # 如果统一保存失败，尝试agent自己的save方法（向后兼容）
                try:
                    # 创建临时目录
                    os.makedirs("models", exist_ok=True)
                    
                    # 调用agent的save方法
                    agent.save()
                    
                    # 移动文件到目标目录
                    patterns = [f"models/{idx}_agent.*", f"models/{idx}.*"]
                    moved = False
                    for pat in patterns:
                        for src_path in glob.glob(pat):
                            dst_path = os.path.join(model_dir, os.path.basename(src_path))
                            try:
                                shutil.move(src_path, dst_path)
                                print(f"  [saved] {idx} → {dst_path} (legacy format)")
                                moved = True
                            except Exception as e:
                                try:
                                    shutil.copy2(src_path, dst_path)
                                    print(f"  [saved] {idx} → {dst_path} (legacy format, copied)")
                                    moved = True
                                except Exception:
                                    pass
                    
                    if not moved:
                        print(f"  [warn] Could not save {idx} (no files found)")
                        
                except AttributeError:
                    # agent没有save方法，已经尝试了统一保存
                    print(f"  [warn] Could not save {idx} (no save method)")
                except Exception as e:
                    print(f"  [warn] Failed to save {idx}: {e}")
        
        # 清理临时目录
        try:
            if os.path.exists("models"):
                # 清理剩余文件
                for item in os.listdir("models"):
                    try:
                        os.remove(os.path.join("models", item))
                    except Exception:
                        pass
                # 删除目录
                if not os.listdir("models"):
                    os.rmdir("models")
        except Exception:
            pass
    
    def run(self, rounds: int = 500) -> pd.DataFrame:
        """运行测试"""
        n_agents = len(self.agents)
        pairings = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
        total_games = n_agents * (n_agents - 1) * self.games_per_pair * rounds
        
        print(f"\n🎯 Testing with seed {self.seed}")
        print(f"  Agents: {n_agents}")
        print(f"  Total games: {total_games:,}")
        print(f"  Lipschitz analysis: {'one-hot' if self.use_onehot else f'window-{self.history_k}'}")
        
        # 初始化记录
        record_data = []
        lipschitz_data = []
        
        # 初始化分数
        scores = defaultdict(int)
        wins = defaultdict(int)
        losses = defaultdict(int)
        draws = defaultdict(int)
        
        pbar = tqdm(total=total_games, desc=f"Seed {self.seed}")
        
        for r in range(1, rounds + 1):
            for pair_idx, (i, j) in enumerate(pairings, start=1):
                who = self.agents[i]
                whom = self.agents[j]
                pair_key = (who.idxname, whom.idxname)
                
                # 获取动作
                a_who = int(who.punches(r))
                a_whom = int(whom.punches(r))
                
                # Lipschitz分析
                if self.use_onehot:
                    p_true = compute_onehot_distribution(a_whom)
                else:
                    p_true = compute_empirical_distribution(
                        self.opponent_histories[pair_key], 
                        k=self.history_k
                    )
                
                # 获取预测分布
                p_pred, pred_source = extract_prediction_distribution(
                    who, r, 
                    debug=self.debug and r % 100 == 0,
                    allow_hist_fallback=False
                )
                
                # 计算指标
                l1_dist = compute_l1_distance(p_true, p_pred)

                # (1) Played-action regret: how suboptimal the actually played action is
                #     (This can violate the Lipschitz bound if the agent explores or
                #      does not best-respond to its own predicted distribution.)
                regret_played = compute_regret(a_who, p_true)

                # (2) Theory-consistent regret for the Lipschitz bound:
                #     play the best-response to the *predicted* distribution p_pred
                #     (even if the agent itself uses a heuristic decision rule)
                a_br_pred = best_response_action(p_pred)
                regret = compute_regret(a_br_pred, p_true)
                
                # 判断胜负
                delta, winner = judge(a_who, a_whom)
                
                # 更新策略
                try:
                    who.play(a_who, a_whom)
                except Exception:
                    pass
                try:
                    whom.play(a_whom, a_who)
                except Exception:
                    pass
                
                # 更新分数
                scores[who.idxname] += delta
                scores[whom.idxname] -= delta
                
                if winner == "who":
                    wins[who.idxname] += 1
                    losses[whom.idxname] += 1
                elif winner == "whom":
                    wins[whom.idxname] += 1
                    losses[who.idxname] += 1
                else:
                    draws[who.idxname] += 1
                    draws[whom.idxname] += 1
                
                # 记录数据
                record_data.append({
                    "round": r,
                    "pair_index": pair_idx,
                    "who_seat": i + 1,
                    "who_agent": who.idxname,
                    "whom_seat": j + 1,
                    "whom_agent": whom.idxname,
                    "who_choice": action_to_name(a_who),
                    "whom_choice": action_to_name(a_whom),
                    "winner": winner,
                    "score_delta_who": delta,
                })
                
                # Lipschitz数据（跳过预热期）
                if r > self.warmup_rounds and pred_source in ["policy", "stat_counts", "markov", "ml_proba"]:
                    lipschitz_data.append({
                        "seed": self.seed,
                        "round": r,
                        "who_agent": who.idxname,
"whom_agent": whom.idxname,

# actions
"action": int(a_who),
"opponent_action": int(a_whom),
"action_br_pred": int(a_br_pred),
"br_match": int(a_who == a_br_pred),

# distances / regrets
"l1_distance": float(l1_dist),

# NOTE: 'regret' is the theory-consistent regret used in the Lipschitz bound check
#       (best-response to p_pred, evaluated under p_true).
"regret": float(regret),

# extra: the regret of the actually played action (may violate the bound)
"regret_played": float(regret_played),

"pred_source": pred_source,

# distributions
"p_true_rock": float(p_true[0]),
"p_true_paper": float(p_true[1]),
"p_true_scissors": float(p_true[2]),
"p_pred_rock": float(p_pred[0]),
"p_pred_paper": float(p_pred[1]),
"p_pred_scissors": float(p_pred[2]),
                    })
                
                # 更新历史
                self.opponent_histories[pair_key].append(a_whom)
                
                pbar.update(self.games_per_pair)
        
        pbar.close()
        
        # 创建汇总DataFrame
        summary_data = []
        for agent_name in scores.keys():
            summary_data.append({
                "agent": agent_name,
                "score": scores[agent_name],
                "wins": wins[agent_name],
                "losses": losses[agent_name],
                "draws": draws[agent_name],
                "win_rate": wins[agent_name] / (wins[agent_name] + losses[agent_name] + draws[agent_name])
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values("score", ascending=False)
        
        # 保存所有输出
        self._save_all_outputs(record_data, summary_df, lipschitz_data)
        
        # 保存模型
        self._save_models()
        
        return summary_df
    
    def _save_all_outputs(self, record_data, summary_df, lipschitz_data):
        """保存所有输出文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 保存详细记录 (RPS_record_seed*.csv)
        record_df = pd.DataFrame(record_data)
        record_path = os.path.join(self.output_dir, f"RPS_record_seed{self.seed}.csv")
        record_df.to_csv(record_path, index=False)
        print(f"  [saved] {record_path}")
        
        # 2. 保存汇总 (RPS_train_summary_seed*.csv)
        summary_path = os.path.join(self.output_dir, f"RPS_train_summary_seed{self.seed}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"  [saved] {summary_path}")
        
        # 3. 保存Lipschitz数据
        if lipschitz_data:
            lipschitz_dir = os.path.join(self.output_dir, "lipschitz_analysis")
            os.makedirs(lipschitz_dir, exist_ok=True)
            
            lipschitz_df = pd.DataFrame(lipschitz_data)
            lipschitz_path = os.path.join(lipschitz_dir, f"lipschitz_seed{self.seed}.csv")
            lipschitz_df.to_csv(lipschitz_path, index=False)
            print(f"  [saved] {lipschitz_path}")


# ==================
# 多种子管理器
# ==================
class MultiSeedTesterV4:
    """V4版本多种子测试管理器"""
    
    def __init__(self, seeds: List[int], test_seats: List[TestSeatCfg], 
                 input_dir: str, output_dir: str, **kwargs):
        self.seeds = seeds
        self.test_seats = test_seats
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.test_kwargs = kwargs
    
    def run(self, rounds: int = 500) -> pd.DataFrame:
        """运行所有种子的测试"""
        print("\n" + "🎲 " * 20)
        print(f"Multi-seed Testing: {len(self.seeds)} seeds")
        print(f"Seeds: {self.seeds}")
        print("🎲 " * 20)
        
        all_summaries = []
        
        for seed in self.seeds:
            print(f"\n{'='*60}")
            print(f"SEED {seed}")
            print('='*60)
            
            tester = TestSystemV4(
                test_seats=self.test_seats,
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                seed=seed,
                **self.test_kwargs
            )
            
            summary = tester.run(rounds=rounds)
            summary['seed'] = seed  # 添加种子列
            all_summaries.append(summary)
        
        # 合并所有种子的结果
        if all_summaries:
            # 生成RPS_train_all_seeds.csv
            all_df = pd.concat(all_summaries, ignore_index=True)
            all_path = os.path.join(self.output_dir, "RPS_train_all_seeds.csv")
            all_df.to_csv(all_path, index=False)
            print(f"\n✅ All seeds data saved to: {all_path}")
            
            # 生成RPS_train_statistics.csv
            stats = all_df.groupby("agent")["score"].agg(
                ["count", "mean", "std", "min", "max", "median"]
            )
            
            # 添加95% CI
            se = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
            stats["ci95"] = 1.96 * se
            stats["lower_bound"] = stats["mean"] - stats["ci95"]
            stats["upper_bound"] = stats["mean"] + stats["ci95"]
            
            stats = stats.sort_values(by="mean", ascending=False)
            
            stats_path = os.path.join(self.output_dir, "RPS_train_statistics.csv")
            stats.to_csv(stats_path, encoding="utf-8")
            print(f"✅ Statistics saved to: {stats_path}")
            
            # 生成experiment_metadata.json
            metadata = {
                "seeds": self.seeds,
                "n_seeds": len(self.seeds),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "train_kwargs": {
                    "seats_csv": self.test_kwargs.get("seats_csv", "test_agent_seats.csv"),
                    "games_per_pair": self.test_kwargs.get("games_per_pair", 1),
                    "batch_size": self.test_kwargs.get("batch_size", 32),
                    "batch_update_freq": self.test_kwargs.get("batch_freq", 100),
                    "rounds": rounds,
                    "output_dir": self.output_dir,
                    "history_k": self.test_kwargs.get("history_k", 20),
                    "use_onehot": self.test_kwargs.get("use_onehot", False),
                    "warmup": self.test_kwargs.get("warmup_rounds", 50),
                },
            }
            
            metadata_path = os.path.join(self.output_dir, "experiment_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✅ Metadata saved to: {metadata_path}")
            
            # 打印统计摘要
            print("\n📊 Cross-seed Statistics:")
            print(stats[["mean", "std", "min", "max"]].to_string())
            
            # 稳定性分析
            print("\n🎯 Stability Analysis (CV = std/mean):")
            cv = (stats["std"] / stats["mean"].abs().clip(lower=1e-9) * 100).sort_values()
            for agent, value in cv.items():
                if np.isfinite(value):
                    stability = "very stable" if value < 10 else "stable" if value < 20 else "unstable"
                    print(f"  {agent}: CV={value:.1f}% ({stability})")
            
            return stats
        
        return pd.DataFrame()


# ==================
# 主函数
# ==================
def main():
    parser = argparse.ArgumentParser(
        description="RPS v4.0 Complete Testing System with Lipschitz Analysis"
    )
    
    # 基本参数
    parser.add_argument("--rounds", type=int, default=500, 
                       help="Number of rounds per pair")
    parser.add_argument("--games-per-pair", type=int, default=1,
                       help="Games per pair per round")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for neural models")
    parser.add_argument("--batch-freq", type=int, default=32,
                       help="Batch update frequency")
    
    # 种子和座位
    parser.add_argument("--seeds", type=str, default="1,2,3",
                       help="Test seeds (comma-separated)")
    parser.add_argument("--seats", type=str, required=True,
                       help="Seat configuration file")
    
    # 目录
    parser.add_argument("--input-dir", type=str, default="RPS_train_summary",
                       help="Input directory with trained models")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    
    # Lipschitz参数
    parser.add_argument("--history-k", type=int, default=20,
                       help="History window for empirical distribution")
    parser.add_argument("--use-onehot", action="store_true",
                       help="Use one-hot distribution for p_true")
    parser.add_argument("--warmup", type=int, default=50,
                       help="Warmup rounds to skip")
    
    # 调试
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # 解析种子
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    # 加载座位配置
    print(f"\n📋 Loading seats configuration: {args.seats}")
    test_seats = load_test_seats(args.seats)
    print(f"Loaded {len(test_seats)} seats")
    for cfg in test_seats:
        train_info = f"seed={cfg.train_seed}" if cfg.train_seed is not None else "untrained"
        print(f"  Seat {cfg.seat}: {cfg.agent_full} ({cfg.method}) [{train_info}]")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行多种子测试
    tester = MultiSeedTesterV4(
        seeds=seeds,
        test_seats=test_seats,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seats_csv=args.seats,  # 保存到metadata
        games_per_pair=args.games_per_pair,
        batch_size=args.batch_size,
        batch_freq=args.batch_freq,
        history_k=args.history_k,
        use_onehot=args.use_onehot,
        warmup_rounds=args.warmup,
        debug=args.debug
    )
    
    stats = tester.run(rounds=args.rounds)
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE")
    print("="*60)
    print(f"\n📁 All results saved to: {args.output_dir}")
    print("\nKey files generated:")
    print(f"  • Model files: models_seed*/")
    print(f"  • Metadata: experiment_metadata.json")
    print(f"  • Statistics: RPS_train_statistics.csv")
    print(f"  • All seeds: RPS_train_all_seeds.csv")
    print(f"  • Seed summaries: RPS_train_summary_seed*.csv")
    print(f"  • Detailed records: RPS_record_seed*.csv")
    print(f"  • Lipschitz data: lipschitz_analysis/lipschitz_seed*.csv")
    
    print("\n📊 Ready for analysis with:")
    print(f"  • python analyze_multi_seed_1.py --input-dir {args.output_dir} --output-dir analysis_1")
    print(f"  • python analyze_multi_seed_2.py --input-dir {args.output_dir} --out-dir analysis_2")
    print(f"  • python analyze_multi_seed_Lipschitz.py --input-dir {args.output_dir}")
    
    # 清理临时models目录
    if os.path.exists("models") and not os.listdir("models"):
        try:
            os.rmdir("models")
        except Exception:
            pass


if __name__ == "__main__":
    main()