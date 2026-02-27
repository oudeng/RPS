#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quick_gpu_test.py - 快速测试不同batch配置的GPU利用率
"""

import os
import sys
import subprocess
import time
import threading
from datetime import datetime


def monitor_gpu(duration=30, interval=1):
    """监控GPU使用率"""
    gpu_stats = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            stats = result.stdout.strip().split(', ')
            gpu_util = float(stats[0])
            mem_used = float(stats[1])
            power = float(stats[2])
            
            gpu_stats.append({
                'time': time.time() - start_time,
                'gpu_util': gpu_util,
                'mem_used': mem_used,
                'power': power
            })
            
            print(f"\r[{int(time.time()-start_time):3d}s] GPU: {gpu_util:5.1f}% | "
                  f"Mem: {mem_used:6.0f}MB | Power: {power:5.1f}W", end='')
            
        except Exception as e:
            print(f"\nError monitoring GPU: {e}")
        
        time.sleep(interval)
    
    print()  # 新行
    return gpu_stats


def run_test(batch_size, batch_freq, rounds=5, seeds="42"):
    """运行测试"""
    print("\n" + "="*60)
    print(f"测试配置: batch_size={batch_size}, batch_freq={batch_freq}")
    print("="*60)
    
    # 构建命令
    cmd = [
        "python", "train_v3_multi_seed.py",
        "--rounds", str(rounds),
        "--seeds", seeds,
        "--batch-size", str(batch_size),
        "--batch-freq", str(batch_freq),
        "--output-dir", f"test_batch{batch_size}_freq{batch_freq}"
    ]
    
    print(f"命令: {' '.join(cmd)}")
    print("监控GPU中...")
    
    # 启动训练进程
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 监控GPU（30秒）
    stats = monitor_gpu(duration=30)
    
    # 终止进程
    process.terminate()
    try:
        process.wait(timeout=5)
    except:
        process.kill()
    
    # 分析结果
    if stats:
        avg_gpu = sum(s['gpu_util'] for s in stats) / len(stats)
        max_gpu = max(s['gpu_util'] for s in stats)
        avg_mem = sum(s['mem_used'] for s in stats) / len(stats)
        avg_power = sum(s['power'] for s in stats) / len(stats)
        
        print(f"\n📊 统计结果:")
        print(f"  平均GPU利用率: {avg_gpu:.1f}%")
        print(f"  最大GPU利用率: {max_gpu:.1f}%")
        print(f"  平均显存使用: {avg_mem:.0f}MB")
        print(f"  平均功耗: {avg_power:.1f}W")
        
        return {
            'batch_size': batch_size,
            'batch_freq': batch_freq,
            'avg_gpu': avg_gpu,
            'max_gpu': max_gpu,
            'avg_mem': avg_mem,
            'avg_power': avg_power
        }
    
    return None


def main():
    print("\n🚀 GPU利用率快速测试工具")
    print("="*60)
    
    # 测试配置列表
    test_configs = [
        (32, 100),   # 默认
        (64, 64),    # 轻度优化
        (128, 128),  # 推荐
        (256, 256),  # 激进
    ]
    
    results = []
    
    print("\n将测试以下配置:")
    for bs, bf in test_configs:
        print(f"  - batch_size={bs}, batch_freq={bf}")
    
    input("\n按Enter开始测试...")
    
    for batch_size, batch_freq in test_configs:
        result = run_test(batch_size, batch_freq, rounds=5)
        if result:
            results.append(result)
        time.sleep(2)  # 冷却
    
    # 输出汇总
    print("\n" + "="*60)
    print("📊 测试汇总")
    print("="*60)
    print(f"{'Batch':<8} {'Freq':<8} {'GPU%':<8} {'Max%':<8} {'Mem(MB)':<10} {'Power(W)':<10}")
    print("-"*60)
    
    for r in sorted(results, key=lambda x: x['avg_gpu'], reverse=True):
        print(f"{r['batch_size']:<8} {r['batch_freq']:<8} "
              f"{r['avg_gpu']:<8.1f} {r['max_gpu']:<8.1f} "
              f"{r['avg_mem']:<10.0f} {r['avg_power']:<10.1f}")
    
    # 推荐最优配置
    if results:
        best = max(results, key=lambda x: x['avg_gpu'])
        print(f"\n🎯 推荐配置:")
        print(f"python train_v3_multi_seed.py \\")
        print(f"    --rounds 100 \\")
        print(f"    --seeds 1,2,3,5,8 \\")
        print(f"    --batch-size {best['batch_size']} \\")
        print(f"    --batch-freq {best['batch_freq']}")
        print(f"\n预期GPU利用率: {best['avg_gpu']:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试中断")
    except Exception as e:
        print(f"\n错误: {e}")
