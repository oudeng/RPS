#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_multi_seed_Lipschitz_v2.py - 增强版Lipschitz界分析工具
on Jan 10, 2026

新增参数：
--separate-matchups: 为每个对战方向生成独立的图表
--matchup-filter: 只分析特定的对战组合

Usage example:
1. 生成所有对战方向的独立分析
python analyze_multi_seed_Lipschitz_v2.py \
    --input-dir Test_3_1_RNNvsA3C \
    --separate-matchups \
    --palette nature

2. 只分析特定对战
python analyze_multi_seed_Lipschitz_v2.py \
    --input-dir Test_3_1_RNNvsA3C \
    --matchup-filter "RNN,A3C" \
    --separate-matchups

3. 传统混合分析（保留原功能）
python analyze_multi_seed_Lipschitz_v2.py \
    --input-dir Test_3_1_RNNvsA3C

# 输出说明
每个单独的对战图包含6个子图：
- 主散点图：显示Lipschitz界，标题明确说明方向
- L1分布：预测误差分布
- Regret分布：明确标注"Agent A optimal/worst"
- 违反检测：显示对战双方和统计信息
- Binned分析：分段统计
- 相关性分析：2D密度图和相关系数

# 对战矩阵热力图
- 左图：平均regret矩阵（谁对谁的后悔值）
- 右图：最优选择率矩阵（谁对谁的最优率）

# 解读优势
- 不再混淆方向：每个图表清楚标明是谁攻击谁
- 易于对比：可以直接比较RNN→A3C vs A3C→RNN的表现
- 完整统计：CSV汇总所有对战的关键指标
"""

import argparse
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 统计检验
try:
    from scipy.stats import spearmanr, gaussian_kde, pearsonr
    from scipy import stats
    from sklearn.linear_model import LinearRegression, HuberRegressor
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    print("[Warning] scipy/sklearn not found, some analyses will be skipped")

# 可视化
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

# 设置出版质量的图表样式
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11
rcParams['figure.titlesize'] = 16
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
rcParams['axes.linewidth'] = 1.0
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'

# 专业配色方案
PALETTES = {
    'nature': ['#374E55', '#DF8F44', '#00A1D5', '#B24745', '#79AF97', '#6A6599', '#80796B'],
    'science': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133', '#56B4E9', '#F0E442'],
    'npg': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2'],
    'aaas': ['#3B4992', '#EE0000', '#008B45', '#631879', '#008280', '#BB0021', '#5F559B'],
    'jco': ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67', '#8F7700']
}


def set_publication_style(palette='nature'):
    """设置出版质量的图表样式"""
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.15',
        'axes.linewidth': 1.0
    })
    
    colors = PALETTES.get(palette, PALETTES['nature'])
    sns.set_palette(colors)
    return colors


def ensure_dirs(output_dir: str, separate_matchups: bool = False) -> Dict[str, str]:
    """创建输出目录结构"""
    dirs = {
        'main': output_dir,
        'lipschitz': os.path.join(output_dir, 'lipschitz_figures'),
        'comprehensive': os.path.join(output_dir, 'lipschitz_figures', 'comprehensive_analysis')
    }
    
    # 如果需要分离的matchup分析
    if separate_matchups:
        dirs['matchups'] = os.path.join(output_dir, 'lipschitz_figures', 'matchup_analysis')
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def load_lipschitz_data(input_dir: str) -> pd.DataFrame:
    """加载所有Lipschitz数据"""
    lipschitz_dir = os.path.join(input_dir, 'lipschitz_analysis')
    
    if not os.path.exists(lipschitz_dir):
        raise FileNotFoundError(f"Lipschitz analysis directory not found: {lipschitz_dir}")
    
    all_data = []
    
    # 加载所有CSV文件
    for file_name in sorted(os.listdir(lipschitz_dir)):
        if file_name.startswith('lipschitz_seed') and file_name.endswith('.csv'):
            file_path = os.path.join(lipschitz_dir, file_name)
            
            # 提取种子号
            seed_str = file_name.replace('lipschitz_seed', '').replace('.csv', '')
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            
            # 读取数据
            df = pd.read_csv(file_path)
            df['seed'] = seed
            all_data.append(df)
            print(f"  ✓ Loaded {file_name}: {len(df)} records")
    
    if not all_data:
        raise ValueError(f"No lipschitz data files found in {lipschitz_dir}")
    
    # 合并所有数据
    combined = pd.concat(all_data, ignore_index=True)
    
    # 数据清理
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=['l1_distance', 'regret'])
    
    # 添加对战标识（如果不存在）
    if 'matchup_id' not in combined.columns:
        combined['matchup_id'] = combined['who_agent'] + '_vs_' + combined['whom_agent']
    
    return combined


def check_violations(
    data: pd.DataFrame,
    regret_col: str = "regret",
    L: float = 2.0,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Check Lipschitz bound exceedances: regret <= L * |p-q|_1.

    Notes
    -----
    - For the *theory-consistent* bound in our paper, `regret_col` should correspond to the
      regret of the best-response to the predicted distribution \hat{p}_t, evaluated under p_t.
    - If you instead use played-action regret (e.g., `regret_played`), exceedances are expected
      when the agent explores or does not best-respond to its own predicted distribution.
    """
    if regret_col not in data.columns:
        raise KeyError(f"Missing column: {regret_col}")
    if "l1_distance" not in data.columns:
        raise KeyError("Missing column: l1_distance")

    return data[data[regret_col] > L * data["l1_distance"] + tol].copy()


def extract_agent_names(agent_str: str) -> str:
    """从agent字符串提取简洁名称"""
    # 保留完整的agent名称，包括seed_agent_algorithm格式
    # 例如: "1_51_RNN" -> "1_51_RNN"
    # 或 "2_13_A3C_v2" -> "2_13_A3C_v2"
    
    # 如果名称过长（超过20个字符），可以考虑缩短
    if len(agent_str) > 20:
        parts = agent_str.split('_')
        # 如果是标准格式 seed_agentnum_algorithm...
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            # 保留前三个部分（seed_agentnum_algorithm）
            return '_'.join(parts[:3])
        # 否则尝试缩短
        elif len(parts) > 3:
            return '_'.join(parts[:3]) + '...'
    
    # 对于正常长度的名称，返回完整字符串
    return agent_str


def analyze_single_matchup(data: pd.DataFrame, who_agent: str, whom_agent: str, 
                          dirs: Dict[str, str], colors, title_suffix: str = ""):
    """分析单个对战方向并生成图表（修复版：只使用4个子图）"""
    
    # 筛选特定对战的数据
    mask = (data['who_agent'] == who_agent) & (data['whom_agent'] == whom_agent)
    matchup_data = data[mask].copy()
    
    if len(matchup_data) == 0:
        print(f"  [Warning] No data for {who_agent} vs {whom_agent}")
        return
    
    # 简化agent名称用于显示
    who_name = extract_agent_names(who_agent)
    whom_name = extract_agent_names(whom_agent)
    
    # 创建图表（2x2布局，只使用4个子图）
    fig = plt.figure(figsize=(16, 10))
    
    # 添加总标题
    matchup_title = f"{who_name} vs {whom_name}{title_suffix}"
    fig.suptitle(f'Lipschitz Analysis: {matchup_title}', fontsize=16, fontweight='bold', y=0.98)
    
    l1 = matchup_data['l1_distance'].values
    regret = matchup_data['regret'].values
    
    # === 子图1（左上）: 主散点图 ===
    ax = plt.subplot(2, 2, 1)
    
    # 密度着色
    if _HAVE_SCIPY and len(matchup_data) > 1000:
        sample_idx = np.random.choice(len(matchup_data), min(5000, len(matchup_data)), replace=False)
        l1_sample = l1[sample_idx]
        regret_sample = regret[sample_idx]
    else:
        l1_sample = l1
        regret_sample = regret
    
    scatter = ax.scatter(l1_sample, regret_sample, alpha=0.3, s=10, c=regret_sample, cmap='viridis')
    
    # 添加理论界
    x_theory = np.linspace(0, 2, 100)
    y_theory = 2 * x_theory
    ax.plot(x_theory, y_theory, 'r--', linewidth=2, label='Theory: Δ ≤ 2|p-q|₁')
    
    # 添加回归线
    if _HAVE_SCIPY and len(l1) > 30:
        mask_nonzero = l1 > 0.01
        if mask_nonzero.sum() > 10:
            reg = LinearRegression()
            X_reg = l1[mask_nonzero].reshape(-1, 1)
            y_reg = regret[mask_nonzero]
            reg.fit(X_reg, y_reg)
            x_fit = np.linspace(0, 2, 100)
            y_fit = reg.predict(x_fit.reshape(-1, 1))
            ax.plot(x_fit, y_fit, 'g-', linewidth=1.5, alpha=0.7,
                   label=f'Fit: Δ={reg.coef_[0]:.2f}|p-q|₁+{reg.intercept_:.2f}')
    
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Regret $\\Delta_t$')
    ax.set_title(f'Lipschitz Bound ({who_name}→{whom_name})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.1])
    ax.set_ylim([0, 2.1])
    
    # === 子图2（右上）: L1分布 ===
    ax = plt.subplot(2, 2, 2)
    ax.hist(l1, bins=50, alpha=0.7, color=colors[0], edgecolor='black')
    ax.axvline(np.mean(l1), color='r', linestyle='--', linewidth=2, label=f'μ={np.mean(l1):.3f}')
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Frequency')
    ax.set_title('L1 Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === 子图3（左下）: Regret分布 ===
    ax = plt.subplot(2, 2, 3)
    
    # 检测是否是离散值
    unique_regrets = np.unique(regret)
    if len(unique_regrets) <= 10:
        # 离散分布（如RPS的0,1,2）
        regret_counts = [np.sum(regret == val) for val in [0, 1, 2]]
        # 使用特定颜色：绿色（optimal），黄色（中间），红色（worst）
        bar_colors = ['green', 'gold', 'red']
        bars = ax.bar([0, 1, 2], regret_counts, color=bar_colors, 
                      edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for val, count in enumerate(regret_counts):
            if count > 0:
                percentage = 100 * count / len(regret)
                optimal_label = ""
                if val == 0:
                    optimal_label = f"\n{who_name} optimal"
                elif val == 2:
                    optimal_label = f"\n{who_name} worst"
                ax.text(val, count + max(regret_counts)*0.01,
                       f'{count}\n({percentage:.1f}%){optimal_label}',
                       ha='center', fontsize=10)
        
        ax.set_ylim([0, max(regret_counts) * 1.15])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Δ=0', 'Δ=1', 'Δ=2'])
    else:
        ax.hist(regret, bins=50, alpha=0.7, color=colors[1], edgecolor='black')
    
    ax.set_xlabel('Regret $\\Delta_t$')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Regret Distribution ({who_name})')
    ax.grid(True, alpha=0.3)
    
    # === 子图4（右下）: Binned分析 ===
    ax = plt.subplot(2, 2, 4)
    
    l1_bins = np.linspace(0, 2, 21)
    bin_centers = (l1_bins[:-1] + l1_bins[1:]) / 2
    
    mean_regrets = []
    std_regrets = []
    for i in range(len(l1_bins)-1):
        bin_mask = (l1 >= l1_bins[i]) & (l1 < l1_bins[i+1])
        if np.sum(bin_mask) > 5:
            mean_regrets.append(np.mean(regret[bin_mask]))
            std_regrets.append(np.std(regret[bin_mask]))
        else:
            mean_regrets.append(np.nan)
            std_regrets.append(np.nan)
    
    mean_regrets = np.array(mean_regrets)
    std_regrets = np.array(std_regrets)
    
    valid = ~np.isnan(mean_regrets)
    ax.errorbar(bin_centers[valid], mean_regrets[valid], 
               yerr=std_regrets[valid], 
               fmt='o-', linewidth=2, markersize=6, capsize=5, color=colors[2])
    
    ax.plot([0, 2], [0, 4], 'r--', linewidth=2, label='Theory: Δ=2|p-q|₁')    # --- Bound / diagnostics ---
    violations = check_violations(matchup_data, regret_col="regret") if "regret" in matchup_data.columns else pd.DataFrame()
    violation_rate = len(violations) / len(matchup_data) * 100 if len(matchup_data) > 0 else 0

    mean_regret = matchup_data["regret"].mean() if "regret" in matchup_data.columns else np.nan
    # Treat tiny regrets as zero for stability
    optimal_rate = (matchup_data["regret"] <= 1e-12).mean() * 100 if "regret" in matchup_data.columns else np.nan

    # Played-action regret diagnostics (if available)
    played_exceed = None
    played_rate = None
    if "regret_played" in matchup_data.columns:
        played_exceed = check_violations(matchup_data, regret_col="regret_played")
        played_rate = len(played_exceed) / len(matchup_data) * 100 if len(matchup_data) > 0 else 0

    
    info_text = f"{who_name} vs {whom_name}\n"
    if violation_rate == 0:
        info_text += "✅ Theory bound: OK\n"
    else:
        info_text += f"⚠️ Theory exceedances: {len(violations)} ({violation_rate:.2f}%)\n"

    # Played-action exceedances are informative but not necessarily bugs
    if played_rate is not None:
        info_text += f"Played exceedances: {len(played_exceed)} ({played_rate:.2f}%)\n"
    if "br_match" in matchup_data.columns:
        info_text += f"BR-match: {100*matchup_data['br_match'].mean():.1f}%\n"

    info_text += f"Mean regret: {mean_regret:.3f}\n"
    info_text += f"Optimal rate: {optimal_rate:.1f}%"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$ (binned)')
    ax.set_ylabel('Mean Regret ± std')
    ax.set_title('Binned Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.1])
    ax.set_ylim([0, 2.1])
    
    plt.tight_layout()
    
    # 保存图表
    safe_who = who_name.replace('/', '_')
    safe_whom = whom_name.replace('/', '_')
    
    if 'matchups' in dirs:
        output_dir = dirs['matchups']
    else:
        output_dir = dirs['lipschitz']
    
    fig_path = os.path.join(output_dir, f'lipschitz_{safe_who}_vs_{safe_whom}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [Saved] {fig_path}")
    
    return {
        "who": who_agent,
        "whom": whom_agent,
        "n_records": len(matchup_data),
        "mean_regret": matchup_data["regret"].mean() if "regret" in matchup_data.columns else np.nan,
        "optimal_rate": (matchup_data["regret"] <= 1e-12).mean() * 100 if "regret" in matchup_data.columns else np.nan,
        "mean_l1": matchup_data["l1_distance"].mean(),
        # theory (regret-based) exceedances
        "violations": len(violations),
        "violation_rate": violation_rate,
        # optional diagnostics
        "played_exceedances": (len(played_exceed) if played_exceed is not None else np.nan),
        "played_exceedance_rate": (played_rate if played_rate is not None else np.nan),
        "br_match_rate": (100 * matchup_data["br_match"].mean() if "br_match" in matchup_data.columns else np.nan),
    }


def create_comprehensive_plot(data: pd.DataFrame, dirs: Dict[str, str], colors):
    """创建综合分析图（6个子图）并单独保存每个子图"""
    print("\n📊 Comprehensive Lipschitz Analysis")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    
    l1 = data['l1_distance'].values
    regret = data['regret'].values
    
    # 1. 主Lipschitz界散点图
    ax = plt.subplot(2, 3, 1)
    
    # 使用密度着色
    if _HAVE_SCIPY and len(data) > 1000:
        # 下采样用于可视化
        sample_idx = np.random.choice(len(data), min(5000, len(data)), replace=False)
        l1_sample = l1[sample_idx]
        regret_sample = regret[sample_idx]
    else:
        l1_sample = l1
        regret_sample = regret
    
    # 创建hexbin图
    hb = ax.hexbin(l1_sample, regret_sample, gridsize=30, cmap='YlOrRd', 
                   mincnt=1, alpha=0.8)
    plt.colorbar(hb, ax=ax, label='Count')
    
    # 理论界线
    x_theory = np.linspace(0, 2, 200)
    ax.plot(x_theory, 2*x_theory, 'g--', linewidth=2.5, 
           label='Theory: Δ=2|p-q|₁')
    
    # 检查违反
    violations = check_violations(data)
    if len(violations) > 0:
        ax.scatter(violations['l1_distance'], violations['regret'],
                  color='red', s=50, alpha=0.7, marker='x',
                  label=f'Violations ({len(violations)})')
    
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Regret $\\Delta_t$')
    ax.set_title('Lipschitz Bound Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 2.05])
    ax.set_ylim([-0.05, 2.05])
    
    # 2. L1距离分布
    ax = plt.subplot(2, 3, 2)
    ax.hist(l1, bins=50, alpha=0.7, color=colors[0], edgecolor='black')
    ax.axvline(l1.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean={l1.mean():.3f}')
    ax.axvline(np.median(l1), color='blue', linestyle='--', linewidth=2,
              label=f'Median={np.median(l1):.3f}')
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Frequency')
    ax.set_title('L1 Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Regret分布
    ax = plt.subplot(2, 3, 3)
    
    # 检查是否是离散分布
    unique_regrets = np.unique(regret)
    if len(unique_regrets) <= 3:
        # 离散分布（0, 1, 2）
        regret_counts = pd.Series(regret).value_counts().sort_index()
        bars = ax.bar(regret_counts.index, regret_counts.values, 
                     width=0.1, alpha=0.7, edgecolor='black')
        
        # 不同regret值用不同颜色
        color_map = {0: 'green', 1: 'gold', 2: 'red'}
        for bar, val in zip(bars, regret_counts.index):
            bar.set_color(color_map.get(val, 'gray'))
    else:
        # 连续分布
        ax.hist(regret, bins=50, alpha=0.7, color=colors[1], edgecolor='black')
    
    ax.axvline(regret.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean={regret.mean():.3f}')
    ax.set_xlabel('Regret $\\Delta_t$')
    ax.set_ylabel('Frequency')
    ax.set_title('Regret Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 时间演化
    ax = plt.subplot(2, 3, 4)
    
    # 按轮次的移动平均
    if 'round' in data.columns:
        rounds = sorted(data['round'].unique())
        window = max(1, len(rounds) // 50)  # 自适应窗口
        
        l1_by_round = []
        regret_by_round = []
        round_nums = []
        
        for i in range(0, len(rounds), window):
            round_window = rounds[i:i+window]
            mask = data['round'].isin(round_window)
            if mask.sum() > 0:
                l1_by_round.append(data.loc[mask, 'l1_distance'].mean())
                regret_by_round.append(data.loc[mask, 'regret'].mean())
                round_nums.append(np.mean(round_window))
        
        ax2 = ax.twinx()
        line1 = ax.plot(round_nums, l1_by_round, 'b-', linewidth=2, 
                       label='L1 distance', alpha=0.8)
        line2 = ax2.plot(round_nums, regret_by_round, 'r-', linewidth=2,
                        label='Regret', alpha=0.8)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('L1 Distance', color='b')
        ax2.set_ylabel('Regret', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_title('Temporal Evolution')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    # 5. Agent比较
    ax = plt.subplot(2, 3, 5)
    
    # 按agent统计
    agents = data['who_agent'].unique()
    
    if len(agents) <= 10:
        # 少量agents用箱线图
        agent_data = []
        agent_labels = []
        
        for agent in sorted(agents):
            agent_mask = data['who_agent'] == agent
            if agent_mask.sum() > 0:
                # 计算Lipschitz斜率
                l1_agent = data.loc[agent_mask, 'l1_distance'].values
                regret_agent = data.loc[agent_mask, 'regret'].values
                
                # 过滤零点
                nonzero = l1_agent > 0.01
                if nonzero.sum() > 0:
                    slopes = regret_agent[nonzero] / l1_agent[nonzero]
                    agent_data.append(slopes)
                    agent_labels.append(extract_agent_names(agent))
        
        if agent_data:
            bp = ax.boxplot(agent_data, labels=agent_labels, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i % len(colors)])
            ax.axhline(2.0, color='r', linestyle='--', linewidth=2, 
                      label='Theory limit')
            ax.set_ylabel('Lipschitz Slope (Δ/|p-q|₁)')
            ax.set_title('Agent Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        # 多agents用热图
        agent_stats = data.groupby('who_agent').agg({
            'l1_distance': 'mean',
            'regret': 'mean'
        })
        agent_stats['slope'] = agent_stats['regret'] / (agent_stats['l1_distance'] + 1e-6)
        top_agents = agent_stats.nlargest(10, 'slope')
        
        ax.barh(range(len(top_agents)), top_agents['slope'].values, color=colors[0])
        ax.axvline(2.0, color='r', linestyle='--', linewidth=2, label='Theory limit')
        ax.set_yticks(range(len(top_agents)))
        ax.set_yticklabels([extract_agent_names(a) for a in top_agents.index])
        ax.set_xlabel('Mean Lipschitz Slope')
        ax.set_title('Top 10 Agents by Slope')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. 联合分布（2D密度图）
    ax = plt.subplot(2, 3, 6)
    
    if _HAVE_SCIPY and len(data) > 100:
        try:
            # 2D KDE
            mask = (l1 > 0.01) & (regret > 0.01)
            x = l1[mask]
            y = regret[mask]
            
            # 创建网格
            xx, yy = np.mgrid[0:2:100j, 0:2:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel = gaussian_kde(np.vstack([x, y]))
            density = np.reshape(kernel(positions).T, xx.shape)
            
            # 绘制等高线
            contour = ax.contourf(xx, yy, density, levels=10, cmap='YlOrRd', alpha=0.7)
            plt.colorbar(contour, ax=ax, label='Density')
            
            # 添加理论线
            ax.plot([0, 1], [0, 2], 'g--', linewidth=2.5, label='Theory')
            
            ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
            ax.set_ylabel('Regret $\\Delta_t$')
            ax.set_title('Joint Density')
            ax.legend()
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
        except Exception as e:
            # 备用方案：散点图
            ax.scatter(l1, regret, alpha=0.1, s=1)
            ax.plot([0, 1], [0, 2], 'g--', linewidth=2.5, label='Theory')
            ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
            ax.set_ylabel('Regret $\\Delta_t$')
            ax.set_title('Joint Distribution')
            ax.legend()
    else:
        # 数据太少，用简单散点图
        ax.scatter(l1, regret, alpha=0.3, s=5)
        ax.plot([0, 1], [0, 2], 'g--', linewidth=2.5, label='Theory')
        ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
        ax.set_ylabel('Regret $\\Delta_t$')
        ax.set_title('Joint Distribution')
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    # 总标题
    fig.suptitle('Comprehensive Lipschitz Bound Analysis', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存综合图
    fig_path = os.path.join(dirs['lipschitz'], 'comprehensive_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Comprehensive analysis: {fig_path}")
    
    # ========== 单独保存每个子图 ==========
    print("  Saving individual subplots...")
    
    # 1. Lipschitz Bound Analysis
    fig_individual = plt.figure(figsize=(8, 6))
    ax = fig_individual.add_subplot(111)
    
    # 重新创建hexbin图
    if _HAVE_SCIPY and len(data) > 1000:
        sample_idx = np.random.choice(len(data), min(5000, len(data)), replace=False)
        l1_sample = l1[sample_idx]
        regret_sample = regret[sample_idx]
    else:
        l1_sample = l1
        regret_sample = regret
    
    hb = ax.hexbin(l1_sample, regret_sample, gridsize=30, cmap='YlOrRd', 
                   mincnt=1, alpha=0.8)
    plt.colorbar(hb, ax=ax, label='Count')
    ax.plot(x_theory, 2*x_theory, 'g--', linewidth=2.5, label='Theory: Δ=2|p-q|₁')
    
    if len(violations) > 0:
        ax.scatter(violations['l1_distance'], violations['regret'],
                  color='red', s=50, alpha=0.7, marker='x',
                  label=f'Violations ({len(violations)})')
    
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Regret $\\Delta_t$')
    ax.set_title('Lipschitz Bound Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 2.05])
    ax.set_ylim([-0.05, 2.05])
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['comprehensive'], '1_lipschitz_bound.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. L1 Distance Distribution
    fig_individual = plt.figure(figsize=(8, 6))
    ax = fig_individual.add_subplot(111)
    ax.hist(l1, bins=50, alpha=0.7, color=colors[0], edgecolor='black')
    ax.axvline(l1.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean={l1.mean():.3f}')
    ax.axvline(np.median(l1), color='blue', linestyle='--', linewidth=2,
              label=f'Median={np.median(l1):.3f}')
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Frequency')
    ax.set_title('L1 Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['comprehensive'], '2_l1_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Regret Distribution
    fig_individual = plt.figure(figsize=(8, 6))
    ax = fig_individual.add_subplot(111)
    
    if len(unique_regrets) <= 3:
        regret_counts = pd.Series(regret).value_counts().sort_index()
        bars = ax.bar(regret_counts.index, regret_counts.values, 
                     width=0.1, alpha=0.7, edgecolor='black')
        color_map = {0: 'green', 1: 'gold', 2: 'red'}
        for bar, val in zip(bars, regret_counts.index):
            bar.set_color(color_map.get(val, 'gray'))
    else:
        ax.hist(regret, bins=50, alpha=0.7, color=colors[1], edgecolor='black')
    
    ax.axvline(regret.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean={regret.mean():.3f}')
    ax.set_xlabel('Regret $\\Delta_t$')
    ax.set_ylabel('Frequency')
    ax.set_title('Regret Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['comprehensive'], '3_regret_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Temporal Evolution（如果有round数据）
    if 'round' in data.columns and len(round_nums) > 0:
        fig_individual = plt.figure(figsize=(8, 6))
        ax = fig_individual.add_subplot(111)
        ax2 = ax.twinx()
        
        line1 = ax.plot(round_nums, l1_by_round, 'b-', linewidth=2, 
                       label='L1 distance', alpha=0.8)
        line2 = ax2.plot(round_nums, regret_by_round, 'r-', linewidth=2,
                        label='Regret', alpha=0.8)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('L1 Distance', color='b')
        ax2.set_ylabel('Regret', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_title('Temporal Evolution')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['comprehensive'], '4_temporal_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Agent Comparison（保存独立版本）
    if len(agent_data) > 0:
        fig_individual = plt.figure(figsize=(8, 6))
        ax = fig_individual.add_subplot(111)
        
        bp = ax.boxplot(agent_data, labels=agent_labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
        ax.axhline(2.0, color='r', linestyle='--', linewidth=2, 
                  label='Theory limit')
        ax.set_ylabel('Lipschitz Slope (Δ/|p-q|₁)')
        ax.set_title('Agent Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['comprehensive'], '5_agent_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 6. Joint Density（如果有足够数据）
    if _HAVE_SCIPY and len(data) > 100:
        fig_individual = plt.figure(figsize=(8, 6))
        ax = fig_individual.add_subplot(111)
        
        try:
            mask = (l1 > 0.01) & (regret > 0.01)
            x = l1[mask]
            y = regret[mask]
            
            xx, yy = np.mgrid[0:2:100j, 0:2:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel = gaussian_kde(np.vstack([x, y]))
            density = np.reshape(kernel(positions).T, xx.shape)
            
            contour = ax.contourf(xx, yy, density, levels=10, cmap='YlOrRd', alpha=0.7)
            plt.colorbar(contour, ax=ax, label='Density')
            ax.plot([0, 1], [0, 2], 'g--', linewidth=2.5, label='Theory')
            
            ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
            ax.set_ylabel('Regret $\\Delta_t$')
            ax.set_title('Joint Density')
            ax.legend()
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(dirs['comprehensive'], '6_joint_density.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except:
            pass
    
    print("  ✓ Individual subplots saved to comprehensive_analysis/")


def analyze_by_seed(data: pd.DataFrame, dirs: Dict[str, str], colors):
    """按种子分析Lipschitz界"""
    print("\n📊 Analysis by Seed")
    print("="*60)
    
    seeds = sorted(data['seed'].unique())
    n_seeds = len(seeds)
    
    if n_seeds == 1:
        print(f"  Only one seed ({seeds[0]}), skipping seed-wise analysis")
        return
    
    # 创建子图
    n_cols = min(4, n_seeds)
    n_rows = (n_seeds + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    for i, seed in enumerate(seeds):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        seed_data = data[data['seed'] == seed]
        l1 = seed_data['l1_distance'].values
        regret = seed_data['regret'].values
        
        # 散点图
        ax.scatter(l1, regret, alpha=0.3, s=5, color=colors[i % len(colors)])
        
        # 理论线
        x_theory = np.linspace(0, 2, 100)
        ax.plot(x_theory, 2*x_theory, 'r--', linewidth=1.5, 
               alpha=0.7, label='Theory')
        
        # 违反检测
        violations = check_violations(seed_data)
        violation_rate = len(violations) / len(seed_data) * 100
        
        title = f'Seed {seed} (n={len(seed_data)})'
        if violation_rate > 0:
            title += f'\n⚠️ {violation_rate:.1f}% violations'
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('$|p-q|_1$', fontsize=9)
        ax.set_ylabel('Regret', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Lipschitz Analysis by Seed', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig_path = os.path.join(dirs['lipschitz'], 'analysis_by_seed.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Seed analysis: {fig_path}")


def analyze_distribution_details(data: pd.DataFrame, dirs: Dict[str, str], colors):
    """分析分布详情（不含Q-Q plots）"""
    print("\n📊 Distribution Analysis")
    print("="*60)
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. L1距离的详细分布
    ax = plt.subplot(2, 3, 1)
    ax.hist(data['l1_distance'], bins=100, alpha=0.7, 
           color=colors[0], edgecolor='black', density=True)
    
    # 添加KDE曲线
    if _HAVE_SCIPY and len(data) > 100:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data['l1_distance'])
        x_kde = np.linspace(0, 2, 200)
        ax.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')
    
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Density')
    ax.set_title('L1 Distance Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Regret的详细分布
    ax = plt.subplot(2, 3, 2)
    
    unique_regrets = data['regret'].unique()
    if len(unique_regrets) <= 3:
        # 离散分布
        regret_counts = data['regret'].value_counts().sort_index()
        total = len(data)
        
        bars = ax.bar(regret_counts.index, regret_counts.values / total, 
                      width=0.1, alpha=0.7, edgecolor='black')
        
        # 添加百分比标签
        for bar, (val, count) in zip(bars, regret_counts.items()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{100*count/total:.1f}%', ha='center', va='bottom', fontsize=9)
            
            # 颜色映射
            if val == 0:
                bar.set_color('green')
            elif val == 1:
                bar.set_color('orange')
            elif val == 2:
                bar.set_color('red')
    else:
        ax.hist(data['regret'], bins=50, alpha=0.7,
               color=colors[1], edgecolor='black', density=True)
    
    ax.set_xlabel('Regret $\\Delta_t$')
    ax.set_ylabel('Density/Probability')
    ax.set_title('Regret Distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. 条件分布（按L1范围）
    ax = plt.subplot(2, 3, 3)
    
    # 创建L1的bins
    l1_ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    
    for i, (l1_min, l1_max) in enumerate(l1_ranges):
        mask = (data['l1_distance'] >= l1_min) & (data['l1_distance'] < l1_max)
        if mask.sum() > 10:
            regret_subset = data.loc[mask, 'regret']
            ax.hist(regret_subset, bins=20, alpha=0.5, 
                   label=f'L1∈[{l1_min},{l1_max})', 
                   color=colors[i % len(colors)])
    
    ax.set_xlabel('Regret $\\Delta_t$')
    ax.set_ylabel('Count')
    ax.set_title('Conditional Regret Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 相关性散点（密度着色）
    ax = plt.subplot(2, 3, 4)
    
    # 使用hexbin显示密度
    hb = ax.hexbin(data['l1_distance'], data['regret'], 
                   gridsize=25, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')
    
    # 添加理论线和拟合线
    x_line = np.linspace(0, 2, 100)
    ax.plot(x_line, 2*x_line, 'g--', linewidth=2, label='Theory')
    
    # 拟合线
    if _HAVE_SCIPY:
        mask_nonzero = data['l1_distance'] > 0.01
        if mask_nonzero.sum() > 10:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            X = data.loc[mask_nonzero, 'l1_distance'].values.reshape(-1, 1)
            y = data.loc[mask_nonzero, 'regret'].values
            reg.fit(X, y)
            y_fit = reg.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_fit, 'b-', linewidth=2, 
                   label=f'Fit: slope={reg.coef_[0]:.2f}')
    
    ax.set_xlabel('$|p_t - \\hat{p}_t|_1$')
    ax.set_ylabel('Regret $\\Delta_t$')
    ax.set_title('Density Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 统计摘要
    ax = plt.subplot(2, 3, 5)
    ax.axis('off')
    
    # 计算统计量
    l1_stats = {
        'Mean': data['l1_distance'].mean(),
        'Std': data['l1_distance'].std(),
        'Min': data['l1_distance'].min(),
        'Q1': data['l1_distance'].quantile(0.25),
        'Median': data['l1_distance'].median(),
        'Q3': data['l1_distance'].quantile(0.75),
        'Max': data['l1_distance'].max()
    }
    
    regret_stats = {
        'Mean': data['regret'].mean(),
        'Std': data['regret'].std(),
        'Min': data['regret'].min(),
        'Q1': data['regret'].quantile(0.25),
        'Median': data['regret'].median(),
        'Q3': data['regret'].quantile(0.75),
        'Max': data['regret'].max()
    }
    
    # 创建表格
    stats_text = "L1 Distance Statistics:\n"
    stats_text += "-" * 30 + "\n"
    for key, val in l1_stats.items():
        stats_text += f"{key:8s}: {val:8.4f}\n"
    
    stats_text += "\nRegret Statistics:\n"
    stats_text += "-" * 30 + "\n"
    for key, val in regret_stats.items():
        stats_text += f"{key:8s}: {val:8.4f}\n"
    
    # 添加违反统计
    violations = check_violations(data)
    stats_text += "\nViolation Analysis:\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"Total: {len(violations)}\n"
    stats_text += f"Rate: {100*len(violations)/len(data):.2f}%\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax.set_title('Statistical Summary')
    
    # 6. CDF比较
    ax = plt.subplot(2, 3, 6)
    
    # 计算经验CDF
    l1_sorted = np.sort(data['l1_distance'])
    l1_cdf = np.arange(1, len(l1_sorted) + 1) / len(l1_sorted)
    
    regret_sorted = np.sort(data['regret'])
    regret_cdf = np.arange(1, len(regret_sorted) + 1) / len(regret_sorted)
    
    ax.plot(l1_sorted, l1_cdf, 'b-', linewidth=2, label='L1 Distance')
    ax.plot(regret_sorted, regret_cdf, 'r-', linewidth=2, label='Regret')
    
    # 理论CDF（如果是均匀分布）
    x_uniform = np.linspace(0, 2, 100)
    y_uniform = x_uniform / 2
    ax.plot(x_uniform, y_uniform, 'g--', linewidth=1.5, 
           label='Uniform[0,2]', alpha=0.7)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution Details', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(dirs['lipschitz'], 'distribution_details.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Distribution details: {fig_path}")


def create_matchup_matrix(data: pd.DataFrame, dirs: Dict[str, str], colors):
    """创建对战矩阵热图"""
    print("\n📊 Creating Matchup Matrix")
    print("="*60)
    
    # 获取所有unique agents
    all_agents = sorted(set(data['who_agent'].unique()) | set(data['whom_agent'].unique()))
    n_agents = len(all_agents)
    
    # 创建矩阵
    mean_regret_matrix = np.zeros((n_agents, n_agents))
    optimal_rate_matrix = np.zeros((n_agents, n_agents))
    
    for i, who in enumerate(all_agents):
        for j, whom in enumerate(all_agents):
            if i != j:  # 跳过自己对自己
                mask = (data['who_agent'] == who) & (data['whom_agent'] == whom)
                if mask.sum() > 0:
                    mean_regret_matrix[i, j] = data.loc[mask, 'regret'].mean()
                    optimal_rate_matrix[i, j] = (data.loc[mask, 'regret'] == 0).mean() * 100
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 简化agent名称用于显示
    agent_labels = [extract_agent_names(a) for a in all_agents]
    
    # 1. Mean Regret热图
    im1 = axes[0].imshow(mean_regret_matrix, cmap='RdYlGn_r', vmin=0, vmax=2)
    axes[0].set_xticks(range(n_agents))
    axes[0].set_yticks(range(n_agents))
    axes[0].set_xticklabels(agent_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(agent_labels)
    axes[0].set_xlabel('Opponent (whom)')
    axes[0].set_ylabel('Player (who)')
    axes[0].set_title('Mean Regret by Matchup')
    plt.colorbar(im1, ax=axes[0], label='Mean Regret')
    
    # 2. Optimal Rate热图
    im2 = axes[1].imshow(optimal_rate_matrix, cmap='RdYlGn', vmin=0, vmax=100)
    axes[1].set_xticks(range(n_agents))
    axes[1].set_yticks(range(n_agents))
    axes[1].set_xticklabels(agent_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(agent_labels)
    axes[1].set_xlabel('Opponent (whom)')
    axes[1].set_ylabel('Player (who)')
    axes[1].set_title('Optimal Rate (%) by Matchup')
    plt.colorbar(im2, ax=axes[1], label='Optimal Rate (%)')
    
    plt.suptitle('Matchup Performance Matrix', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(dirs['lipschitz'], 'matchup_matrix.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Matchup matrix: {fig_path}")


def analyze_all_matchups(data: pd.DataFrame, dirs: Dict[str, str], colors):
    """分析所有对战组合"""
    print("\n📊 Analyzing All Matchups")
    print("="*60)
    
    # 获取所有唯一的对战组合
    matchups = data.groupby(['who_agent', 'whom_agent']).size().reset_index(name='count')
    n_matchups = len(matchups)
    
    print(f"Found {n_matchups} unique matchups")
    
    # 分析每个对战
    results = []
    for _, row in matchups.iterrows():
        who = row['who_agent']
        whom = row['whom_agent']
        
        result = analyze_single_matchup(data, who, whom, dirs, colors)
        if result:
            results.append(result)
    
    # 创建汇总表格
    if results:
        summary_df = pd.DataFrame(results)
        
        # 保存到CSV
        csv_path = os.path.join(dirs['lipschitz'], 'matchup_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\n[Saved] Matchup summary: {csv_path}")
        
        # 打印统计
        print("\n" + "="*60)
        print("MATCHUP STATISTICS")
        print("="*60)
        
        # 找出最佳和最差的matchups
        best_matchups = summary_df.nsmallest(5, 'mean_regret')
        worst_matchups = summary_df.nlargest(5, 'mean_regret')
        
        print("\nTop 5 Best Matchups (lowest mean regret):")
        for _, row in best_matchups.iterrows():
            who_name = extract_agent_names(row['who'])
            whom_name = extract_agent_names(row['whom'])
            print(f"  {who_name} vs {whom_name}: {row['mean_regret']:.3f}")
        
        print("\nTop 5 Worst Matchups (highest mean regret):")
        for _, row in worst_matchups.iterrows():
            who_name = extract_agent_names(row['who'])
            whom_name = extract_agent_names(row['whom'])
            print(f"  {who_name} vs {whom_name}: {row['mean_regret']:.3f}")
        
        # 违反检测汇总
        total_violations = summary_df['violations'].sum()
        matchups_with_violations = (summary_df['violations'] > 0).sum()
        
        print(f"\nViolation Summary:")
        print(f"  Total violations: {total_violations}")
        print(f"  Matchups with violations: {matchups_with_violations}/{n_matchups}")
        
        if matchups_with_violations > 0:
            print("\n  Matchups with violations:")
            violating_matchups = summary_df[summary_df['violations'] > 0].sort_values('violations', ascending=False)
            for _, row in violating_matchups.head(10).iterrows():
                who_name = extract_agent_names(row['who'])
                whom_name = extract_agent_names(row['whom'])
                print(f"    {who_name} vs {whom_name}: {row['violations']} violations ({row['violation_rate']:.2f}%)")


def analyze_data_diagnostics(data: pd.DataFrame) -> Dict:
    """数据诊断分析"""
    print("\n🔍 Data Diagnostics")
    print("="*60)
    
    diagnostics = {}
    
    # 基本统计
    print("\n1. Basic Statistics:")
    print(f"   Total records: {len(data):,}")
    
    seeds = sorted(data['seed'].unique())
    print(f"   Unique seeds: {seeds}")
    
    agents = sorted(data['who_agent'].unique())
    print(f"   Unique agents: {len(agents)}")
    
    rounds = data['round'].unique()
    print(f"   Rounds range: [{rounds.min()}, {rounds.max()}]")
    
    # 预测来源分析
    print("\n2. Prediction Sources:")
    pred_sources = data['pred_source'].value_counts()
    for source, count in pred_sources.items():
        print(f"   {source}: {count:,} ({100*count/len(data):.1f}%)")
    
    # 检查理论界
    print(f"\n3. Theoretical Bound Check (Δ ≤ 2|p-q|₁):")

    # Always define for downstream summary
    violations = pd.DataFrame()

    # Prefer theory-consistent regret if available
    if "regret" in data.columns:
        violations = check_violations(data, regret_col="regret")
        if len(violations) == 0:
            print("   ✅ All points satisfy theoretical bound (checked on 'regret')")
        else:
            print(f"   ⚠️ EXCEEDANCES DETECTED on 'regret': {len(violations)} ({100*len(violations)/len(data):.2f}%)")
            print("      Unexpected for theory-consistent regret; check logging/scaling.")
    elif "regret_played" in data.columns:
        # Legacy dataset: only played-action regret exists
        violations = check_violations(data, regret_col="regret_played")
        if len(violations) == 0:
            print("   ✅ No exceedances (checked on legacy 'regret_played')")
        else:
            print(f"   ⚠️ Exceedances on legacy 'regret_played': {len(violations)} ({100*len(violations)/len(data):.2f}%)")
            print("      This can be normal if the agent does not best-respond to its own prediction.")
    else:
        print("   ❌ Cannot check bound: missing regret columns ('regret' or 'regret_played').")

    # Extra diagnostic: played-action exceedances (informative; may be >0)
    if "regret_played" in data.columns and "regret" in data.columns:
        played_ex = check_violations(data, regret_col="regret_played")
        rate = 100 * len(played_ex) / len(data) if len(data) > 0 else 0.0
        print(f"   • Played-action exceedances: {len(played_ex)} ({rate:.2f}%)")
        if "br_match" in data.columns:
            br_rate = 100 * data["br_match"].mean()
            print(f"   • BR-match rate (action == BR(\hat{{p}})): {br_rate:.1f}%")
    
    # 分布类型检测
    print("\n4. Distribution Type Analysis:")
    unique_l1 = len(data['l1_distance'].unique())
    
    if unique_l1 < 50:
        print("   📌 Detected ONE-HOT distribution (discrete L1)")
        print(f"      Unique L1 values: {unique_l1}")
    else:
        print("   📌 Detected WINDOW-based distribution (continuous L1)")
        print(f"      Unique L1 values: {unique_l1}")
    
    # 数据质量指标
    print("\n5. Data Quality Metrics:")
    l1_mean = data['l1_distance'].mean()
    l1_std = data['l1_distance'].std()
    regret_mean = data['regret'].mean()
    regret_std = data['regret'].std()
    
    print(f"   L1 distance: μ={l1_mean:.4f}, σ={l1_std:.4f}")
    print(f"   Regret: μ={regret_mean:.4f}, σ={regret_std:.4f}")
    
    # 检查异常值
    l1_outliers = ((data['l1_distance'] < 0) | (data['l1_distance'] > 2)).sum()
    regret_outliers = ((data['regret'] < 0) | (data['regret'] > 2)).sum()
    
    if l1_outliers > 0:
        print(f"   ⚠️ L1 outliers (outside [0,2]): {l1_outliers}")
    if regret_outliers > 0:
        print(f"   ⚠️ Regret outliers (outside [0,2]): {regret_outliers}")
    
    print("\n" + "-"*50)
    
    diagnostics = {
        'n_records': len(data),
        'n_seeds': len(seeds),
        'n_agents': len(agents),
        'n_violations': len(violations),
        'violation_rate': len(violations) / len(data) if len(data) > 0 else 0,
        'l1_mean': l1_mean,
        'l1_std': l1_std,
        'regret_mean': regret_mean,
        'regret_std': regret_std,
        'unique_l1': unique_l1
    }
    
    return diagnostics


def generate_summary_report(data: pd.DataFrame, dirs: Dict[str, str]):
    """生成文本摘要报告"""
    print("\n📝 Generating Summary Report")
    print("="*60)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("LIPSCHITZ BOUND ANALYSIS REPORT")
    report_lines.append("="*70)
    report_lines.append("")
    
    # 1. 数据概览
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-"*40)
    report_lines.append(f"Total records: {len(data):,}")
    
    seeds = sorted(data['seed'].unique())
    report_lines.append(f"Test seeds: {seeds}")
    
    agents = data['who_agent'].unique()
    report_lines.append(f"Number of agents: {len(agents)}")
    
    rounds = data['round'].unique()
    report_lines.append(f"Rounds analyzed: {rounds.min()}-{rounds.max()}")
    
    # 2. 分布统计
    report_lines.append("")
    report_lines.append("2. DISTRIBUTION STATISTICS")
    report_lines.append("-"*40)
    
    l1_mean = data['l1_distance'].mean()
    l1_std = data['l1_distance'].std()
    l1_min = data['l1_distance'].min()
    l1_max = data['l1_distance'].max()
    l1_q = data['l1_distance'].quantile([0.25, 0.5, 0.75]).values
    
    report_lines.append("L1 Distance |p-q|₁:")
    report_lines.append(f"  Mean ± Std: {l1_mean:.4f} ± {l1_std:.4f}")
    report_lines.append(f"  Min / Max: {l1_min:.4f} / {l1_max:.4f}")
    report_lines.append(f"  Quartiles: {l1_q}")
    
    regret_mean = data['regret'].mean()
    regret_std = data['regret'].std()
    regret_min = data['regret'].min()
    regret_max = data['regret'].max()
    regret_q = data['regret'].quantile([0.25, 0.5, 0.75]).values
    
    report_lines.append("")
    report_lines.append("Regret Δ:")
    report_lines.append(f"  Mean ± Std: {regret_mean:.4f} ± {regret_std:.4f}")
    report_lines.append(f"  Min / Max: {regret_min:.4f} / {regret_max:.4f}")
    report_lines.append(f"  Quartiles: {regret_q}")
    
    # 3. Lipschitz界分析
    report_lines.append("")
    report_lines.append("3. LIPSCHITZ BOUND ANALYSIS")
    report_lines.append("-"*40)
    
    # Bound check summary
    theory_col = "regret" if "regret" in data.columns else None
    played_col = "regret_played" if "regret_played" in data.columns else None

    if theory_col is None and played_col is not None:
        # legacy dataset: only played-action regret available
        theory_col = played_col

    if theory_col is None:
        report_lines.append("❌ Bound check skipped: no regret columns found.")
        violations = pd.DataFrame()
    else:
        violations = check_violations(data, regret_col=theory_col)
        if len(violations) == 0:
            report_lines.append(f"✅ All points satisfy Δ ≤ 2|p-q|₁ (checked on '{theory_col}')")
        else:
            report_lines.append(f"⚠️ EXCEEDANCES DETECTED on '{theory_col}': {len(violations)} ({100*len(violations)/len(data):.2f}%)")
            report_lines.append("  If this is theory-consistent regret (BR(\\hat{p}) under p), exceedances are unexpected.")
            report_lines.append("  If this is played-action regret, exceedances can be normal due to exploration / non-BR actions.")

    # Optional: played-action diagnostics (informative)
    if played_col is not None and played_col != theory_col:
        played_ex = check_violations(data, regret_col=played_col)
        report_lines.append(f"Played-action exceedances on '{played_col}': {len(played_ex)} ({100*len(played_ex)/len(data):.2f}%)")
        if "br_match" in data.columns:
            report_lines.append(f"BR-match rate (action == BR(\\hat{{p}})): {100*data['br_match'].mean():.1f}%")
    
    # 相关性分析
    if _HAVE_SCIPY:
        mask = (data['l1_distance'] > 0.01) & (data['regret'] > 0.01)
        if mask.sum() > 30:
            corr_spearman, p_spearman = spearmanr(
                data.loc[mask, 'l1_distance'], 
                data.loc[mask, 'regret']
            )
            corr_pearson, p_pearson = pearsonr(
                data.loc[mask, 'l1_distance'], 
                data.loc[mask, 'regret']
            )
            
            report_lines.append("")
            report_lines.append("Correlation Analysis (excluding zeros):")
            report_lines.append(f"  Spearman ρ: {corr_spearman:.4f} (p={p_spearman:.2e})")
            report_lines.append(f"  Pearson r: {corr_pearson:.4f} (p={p_pearson:.2e})")
            
            # 线性回归
            X = data.loc[mask, 'l1_distance'].values.reshape(-1, 1)
            y = data.loc[mask, 'regret'].values
            reg = LinearRegression()
            reg.fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r2 = reg.score(X, y)
            
            report_lines.append("")
            report_lines.append("Linear Regression:")
            report_lines.append(f"  Equation: Δ = {slope:.4f} × |p-q|₁ + {intercept:.4f}")
            report_lines.append(f"  R²: {r2:.4f}")
            report_lines.append(f"  Slope vs Theory: {slope:.4f} vs 2.000")
    
    # 4. 预测源分析
    report_lines.append("")
    report_lines.append("4. PREDICTION SOURCE ANALYSIS")
    report_lines.append("-"*40)
    
    pred_sources = data['pred_source'].value_counts()
    for source, count in pred_sources.items():
        report_lines.append(f"{source}: {count:,} ({100*count/len(data):.1f}%)")
    
    # 5. Top Agents
    report_lines.append("")
    report_lines.append("5. TOP AGENTS BY LIPSCHITZ SLOPE")
    report_lines.append("-"*40)
    
    # 计算每个agent的平均斜率
    agent_slopes = {}
    for agent in data['who_agent'].unique():
        agent_mask = data['who_agent'] == agent
        l1_agent = data.loc[agent_mask, 'l1_distance'].values
        regret_agent = data.loc[agent_mask, 'regret'].values
        
        # 过滤零点
        nonzero = l1_agent > 0.01
        if nonzero.sum() > 0:
            slopes = regret_agent[nonzero] / l1_agent[nonzero]
            agent_slopes[agent] = np.median(slopes)  # 使用中位数更稳健
    
    # 排序并显示top agents
    sorted_agents = sorted(agent_slopes.items(), key=lambda x: x[1], reverse=True)
    for agent, slope in sorted_agents[:5]:
        report_lines.append(f"{agent}: slope={slope:.3f}")
    
    # 写入文件
    report_path = os.path.join(dirs['main'], 'lipschitz_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        f.write('\n\n')
        f.write('='*70)
        f.write('\nEND OF REPORT\n')
        f.write('='*70)
    
    print(f"[Saved] Text report: {report_path}")
    
    # 打印报告
    print("\n" + "="*70)
    for line in report_lines:
        print(line)
    print("="*70)
    print("END OF REPORT")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Lipschitz bounds from multi-seed test results with matchup support'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory with test results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis (default: same as input)')
    parser.add_argument('--palette', type=str, default='nature',
                       choices=list(PALETTES.keys()),
                       help='Color palette for plots')
    parser.add_argument('--separate-matchups', action='store_true',
                       help='Generate separate plots for each matchup')
    parser.add_argument('--matchup-filter', type=str, default=None,
                       help='Only analyze specific matchup (e.g., "51_RNN_vs_13_A3C")')
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    print("\n" + "="*70)
    print("🔬 LIPSCHITZ BOUND ANALYSIS (v2)")
    print("="*70)
    
    # 创建目录结构
    dirs = ensure_dirs(output_dir, args.separate_matchups)
    
    # 设置绘图样式
    colors = set_publication_style(args.palette)
    
    # 加载数据
    print(f"\n📂 Loading data...")
    try:
        data = load_lipschitz_data(args.input_dir)
        print(f"\n📊 Total records: {len(data):,}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 数据诊断
    diagnostics = analyze_data_diagnostics(data)
    
    # 创建可视化
    
    # 1. 综合分析图（包括单独保存子图）
    create_comprehensive_plot(data, dirs, colors)
    print("\n" + "-"*50)
    
    # 2. 按种子分析
    analyze_by_seed(data, dirs, colors)
    print("\n" + "-"*50)
    
    # 3. 分布详情（不含Q-Q plots）
    analyze_distribution_details(data, dirs, colors)
    print("\n" + "-"*50)
    
    # 4. 对战分析
    if args.matchup_filter:
        # 只分析特定对战
        parts = args.matchup_filter.split('_vs_')
        if len(parts) == 2:
            who_agent = parts[0]
            whom_agent = parts[1]
            print(f"\n📊 Analyzing specific matchup: {who_agent} vs {whom_agent}")
            analyze_single_matchup(data, who_agent, whom_agent, dirs, colors)
        else:
            print(f"⚠️ Invalid matchup filter format: {args.matchup_filter}")
            print("   Expected format: 'AgentA_vs_AgentB'")
    elif args.separate_matchups:
        # 分析所有对战
        analyze_all_matchups(data, dirs, colors)
        create_matchup_matrix(data, dirs, colors)
    else:
        # 默认：只创建对战矩阵
        create_matchup_matrix(data, dirs, colors)
    
    print("\n" + "-"*50)
    
    # 5. 生成文本报告
    generate_summary_report(data, dirs)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\n📁 All results saved to: {output_dir}")
    
    # 显示关键文件
    print("\nKey files:")
    print(f"  • Comprehensive analysis: {os.path.join(dirs['lipschitz'], 'comprehensive_analysis.png')}")
    print(f"  • Individual subplots: {os.path.join(dirs['comprehensive'], '*.png')}")
    print(f"  • Distribution details: {os.path.join(dirs['lipschitz'], 'distribution_details.png')}")
    print(f"  • Matchup matrix: {os.path.join(dirs['lipschitz'], 'matchup_matrix.png')}")
    print(f"  • Text report: {os.path.join(output_dir, 'lipschitz_report.txt')}")
    
    if args.separate_matchups:
        print(f"  • Individual matchup plots: {os.path.join(dirs['matchups'], '*.png')}")
        print(f"  • Matchup summary: {os.path.join(dirs['lipschitz'], 'matchup_summary.csv')}")
    
    # 如果有违反，特别提醒
    if diagnostics['n_violations'] > 0:
        print(f"\n⚠️ WARNING: {diagnostics['n_violations']} bound exceedances detected (theory column).")

        print("  If you are using the updated logging (theory-consistent regret), this should be ~0.")
        print("  Otherwise, exceedances may be normal when using played-action regret.")


if __name__ == "__main__":
    main()