#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_regret_direction.py - 分析Regret分布的对战方向

用于判断Regret Distribution图中显示的是哪个agent对哪个agent的regret

Usage：

python utility/analyze_regret_direction.py \
    --input-dir Test_3_1_RNNvsA3C

"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import Counter


def analyze_regret_direction(lipschitz_file: str):
    """
    分析Lipschitz数据中的对战方向
    """
    print("\n" + "="*60)
    print("REGRET DIRECTION ANALYSIS")
    print("="*60)
    
    # 读取数据
    df = pd.read_csv(lipschitz_file)
    print(f"\n📊 Loaded {len(df)} records from {lipschitz_file}")
    
    # 获取所有独特的对战组合
    matchups = df[['who_agent', 'whom_agent']].drop_duplicates()
    print(f"\n🎮 Found {len(matchups)} unique matchups:")
    
    for idx, row in matchups.iterrows():
        who = row['who_agent']
        whom = row['whom_agent']
        
        # 筛选这个对战组合的数据
        mask = (df['who_agent'] == who) & (df['whom_agent'] == whom)
        matchup_data = df[mask]
        
        if len(matchup_data) == 0:
            continue
        
        print(f"\n{'='*50}")
        print(f"Matchup: {who} (who) vs {whom} (whom)")
        print(f"Records: {len(matchup_data)}")
        print('-'*50)
        
        # 分析Regret分布
        regret_counts = matchup_data['regret'].value_counts().sort_index()
        total = len(matchup_data)
        
        print("\n📈 Regret Distribution:")
        print(f"   This shows {who}'s regret when playing against {whom}")
        print(f"   (i.e., how much {who} regrets their action choices)")
        print()
        
        for regret_val in [0, 1, 2]:
            count = regret_counts.get(regret_val, 0)
            percentage = 100 * count / total if total > 0 else 0
            
            interpretation = ""
            if regret_val == 0:
                interpretation = f"← {who} made OPTIMAL choice"
            elif regret_val == 1:
                interpretation = f"← {who} made SUBOPTIMAL choice"
            else:  # regret_val == 2
                interpretation = f"← {who} made WORST choice"
            
            print(f"   Δ={regret_val}: {count:4d} ({percentage:5.1f}%) {interpretation}")
        
        # 计算统计
        mean_regret = matchup_data['regret'].mean()
        std_regret = matchup_data['regret'].std()
        print(f"\n   Mean regret: {mean_regret:.3f} ± {std_regret:.3f}")
        
        # 理论随机基线
        random_mean = 2/3  # 理论期望值
        diff = mean_regret - random_mean
        
        if abs(diff) < 0.05:
            performance = "≈ random performance"
        elif diff < 0:
            performance = f"BETTER than random (by {-diff:.3f})"
        else:
            performance = f"WORSE than random (by {diff:.3f})"
        
        print(f"   vs Random baseline (0.667): {performance}")
        
        # L1距离统计
        mean_l1 = matchup_data['l1_distance'].mean()
        std_l1 = matchup_data['l1_distance'].std()
        print(f"\n   L1 Distance: {mean_l1:.3f} ± {std_l1:.3f}")
        
        # 预测来源统计
        if 'pred_source' in matchup_data.columns:
            pred_sources = matchup_data['pred_source'].value_counts()
            print(f"\n   Prediction sources:")
            for source, count in pred_sources.items():
                print(f"      {source}: {count} ({100*count/len(matchup_data):.1f}%)")
    
    # 总体分析
    print(f"\n{'='*50}")
    print("OVERALL SUMMARY")
    print('='*50)
    
    # 找出所有RNN相关的对战
    rnn_as_who = df[df['who_agent'].str.contains('RNN', na=False)]
    rnn_as_whom = df[df['whom_agent'].str.contains('RNN', na=False)]
    
    a3c_as_who = df[df['who_agent'].str.contains('A3C', na=False)]
    a3c_as_whom = df[df['whom_agent'].str.contains('A3C', na=False)]
    
    print("\n📊 Agent Analysis:")
    
    if len(rnn_as_who) > 0:
        print(f"\n   RNN as attacker (who):")
        print(f"      Total records: {len(rnn_as_who)}")
        print(f"      Mean regret: {rnn_as_who['regret'].mean():.3f}")
        print(f"      Optimal rate (Δ=0): {100*(rnn_as_who['regret']==0).mean():.1f}%")
    
    if len(a3c_as_who) > 0:
        print(f"\n   A3C as attacker (who):")
        print(f"      Total records: {len(a3c_as_who)}")
        print(f"      Mean regret: {a3c_as_who['regret'].mean():.3f}")
        print(f"      Optimal rate (Δ=0): {100*(a3c_as_who['regret']==0).mean():.1f}%")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
When you see a Regret Distribution plot:
1. The title or filename usually indicates the matchup
2. The regret ALWAYS belongs to the 'who' agent (attacker)
3. Δ=0 means 'who' made the best possible choice against 'whom'
4. High Δ=0 percentage → 'who' is playing well against 'whom'

Example:
- If analyzing "51_RNN vs 13_A3C_v2":
  - Regret shows how well RNN (who) plays against A3C (whom)
  - Δ=0 at 35.1% means RNN makes optimal choice 35.1% of the time

- If analyzing "13_A3C_v2 vs 51_RNN":  
  - Regret shows how well A3C (who) plays against RNN (whom)
  - This is the OPPOSITE direction!
    """)


def find_lipschitz_files(base_dir: str):
    """查找所有的Lipschitz数据文件"""
    lipschitz_dir = os.path.join(base_dir, 'lipschitz_analysis')
    
    if not os.path.exists(lipschitz_dir):
        print(f"Error: Lipschitz directory not found: {lipschitz_dir}")
        return []
    
    files = []
    for fname in os.listdir(lipschitz_dir):
        if fname.startswith('lipschitz_seed') and fname.endswith('.csv'):
            files.append(os.path.join(lipschitz_dir, fname))
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze regret direction in Lipschitz data'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing test results with lipschitz_analysis folder')
    parser.add_argument('--seed', type=int, default=None,
                       help='Specific seed to analyze (default: all)')
    
    args = parser.parse_args()
    
    # 查找Lipschitz文件
    files = find_lipschitz_files(args.input_dir)
    
    if not files:
        print(f"No Lipschitz data files found in {args.input_dir}")
        return
    
    # 如果指定了种子，只分析该种子
    if args.seed is not None:
        target_file = os.path.join(
            args.input_dir, 
            'lipschitz_analysis', 
            f'lipschitz_seed{args.seed}.csv'
        )
        if target_file in files:
            files = [target_file]
        else:
            print(f"Warning: No data found for seed {args.seed}")
            return
    
    # 分析每个文件
    for fpath in files:
        seed = os.path.basename(fpath).replace('lipschitz_seed', '').replace('.csv', '')
        print(f"\n{'#'*60}")
        print(f"ANALYZING SEED {seed}")
        print('#'*60)
        analyze_regret_direction(fpath)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()