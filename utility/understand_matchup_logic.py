#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
understand_matchup_logic.py - 理解RPS测试系统中的对战配对逻辑

这个脚本清晰地展示了：
1. 座位如何转换成对战配对
2. Regret是谁对谁的
3. 如何从图表判断方向

python utility/understand_matchup_logic.py

"""

def demonstrate_matchup_logic():
    """
    演示对战配对逻辑
    """
    print("\n" + "="*60)
    print("RPS对战配对逻辑说明")
    print("="*60)
    
    # 模拟您的座位配置
    seats = [
        {"seat": 1, "agent": "51_RNN", "seed": 34},
        {"seat": 2, "agent": "13_A3C_v2", "seed": 89}
    ]
    
    print("\n📋 座位配置 (test_3_1_RNNvsA3C.csv):")
    for s in seats:
        print(f"  Seat {s['seat']}: {s['agent']} (seed={s['seed']})")
    
    # 生成配对（与test_v4中的逻辑相同）
    n_agents = len(seats)
    pairings = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
    
    print(f"\n🎮 生成的对战配对 (共{len(pairings)}个):")
    print("  格式: (i, j) → agents[i] vs agents[j]")
    print("        其中 agents[i] = who（攻击方）")
    print("             agents[j] = whom（防守方）")
    print()
    
    for pair_idx, (i, j) in enumerate(pairings, start=1):
        who = seats[i]['agent']
        whom = seats[j]['agent']
        
        print(f"  配对{pair_idx}: ({i}, {j})")
        print(f"    → {who} (who/攻击方) vs {whom} (whom/防守方)")
        print(f"    → Regret衡量的是: {who}的决策质量")
        print(f"    → L1距离衡量的是: {who}对{whom}行为的预测准确度")
        print()
    
    print("-"*60)
    print("\n💡 关键理解:")
    print()
    print("1. 每个座位组合会产生两个不同方向的对战:")
    print("   • RNN vs A3C: RNN是who（攻击方），A3C是whom（防守方）")
    print("   • A3C vs RNN: A3C是who（攻击方），RNN是whom（防守方）")
    print()
    print("2. Regret的含义:")
    print("   • regret = compute_regret(a_who, p_true)")
    print("   • a_who: who的动作")
    print("   • p_true: whom的行为分布")
    print("   • 含义: who针对whom的策略有多少后悔")
    print()
    print("3. 在您看到的图中:")
    print("   • 如果标题是'RNN vs A3C'，则Regret分布显示的是RNN的后悔值")
    print("   • Δ=0 (35.1%) 表示RNN有35.1%的时间做出了最优选择")
    print("   • Δ=2 (36.2%) 表示RNN有36.2%的时间做出了最差选择")
    
    print("\n" + "="*60)
    print("从Lipschitz数据判断方向")
    print("="*60)
    
    print("""
查看lipschitz_seed*.csv文件中的列:
• who_agent: 攻击方（计算其regret）
• whom_agent: 防守方（作为目标）

例如，如果数据行显示:
{
    "who_agent": "51_RNN",
    "whom_agent": "13_A3C_v2",
    "regret": 0,
    "l1_distance": 0.15,
    ...
}

这表示:
→ RNN（who）对A3C（whom）的这次对战
→ RNN做出了最优选择（regret=0）
→ RNN对A3C行为的预测偏差是0.15
""")
    
    print("\n" + "="*60)
    print("实际代码中的判断方法")
    print("="*60)
    
    print("""
# 在test_v4中的核心代码:

for r in range(1, rounds + 1):
    for pair_idx, (i, j) in enumerate(pairings, start=1):
        who = self.agents[i]      # 攻击方
        whom = self.agents[j]     # 防守方
        
        # 获取双方动作
        a_who = who.punches(r)    # who的动作
        a_whom = whom.punches(r)  # whom的动作
        
        # 计算whom的行为分布
        p_true = compute_empirical_distribution(
            opponent_history_of_whom
        )
        
        # 计算who的预测分布
        p_pred = extract_prediction_from_who()
        
        # 计算who的regret
        regret = compute_regret(a_who, p_true)
        
        # 保存时明确标记方向
        lipschitz_data.append({
            "who_agent": who.idxname,    # 谁的regret
            "whom_agent": whom.idxname,  # 对谁
            "regret": regret,            # who的后悔值
            ...
        })
""")
    
    print("\n✅ 总结：")
    print("• Regret Distribution图显示的是who_agent的后悔分布")
    print("• 查看数据文件的who_agent列可确定是谁的regret")
    print("• 您的图显示的是51_RNN对13_A3C_v2的regret分布")


if __name__ == "__main__":
    demonstrate_matchup_logic()