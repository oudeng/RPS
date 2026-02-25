#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_complete_analysis.py - 完整的测试和分析流程示例

这个脚本展示如何：
1. 运行test_v4进行多种子测试
2. 运行三个分析脚本进行结果分析
3. 整合所有输出

Usage:
    python run_complete_analysis.py
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout:
                print("Output preview:")
                lines = result.stdout.split('\n')[:10]  # 只显示前10行
                for line in lines:
                    print(f"  {line}")
        else:
            print(f"❌ Failed: {description}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete RPS test and analysis pipeline"
    )
    
    # 测试参数
    parser.add_argument("--rounds", type=int, default=500,
                       help="Number of rounds for testing")
    parser.add_argument("--seeds", type=str, default="1,2,3,5,8",
                       help="Test seeds (comma-separated)")
    parser.add_argument("--seats", type=str, required=True,
                       help="Seat configuration file")
    parser.add_argument("--input-dir", type=str, default="RPS_train_summary",
                       help="Directory with trained models")
    
    # 输出目录
    parser.add_argument("--test-dir", type=str, default=None,
                       help="Output directory for test results")
    parser.add_argument("--analysis-dir", type=str, default=None,
                       help="Base directory for analysis outputs")
    
    # Lipschitz参数
    parser.add_argument("--history-k", type=int, default=20,
                       help="History window for Lipschitz analysis")
    parser.add_argument("--use-onehot", action="store_true",
                       help="Use one-hot distribution")
    parser.add_argument("--warmup", type=int, default=50,
                       help="Warmup rounds")
    
    # 分析选项
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip testing, only run analysis")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip analysis, only run testing")
    
    args = parser.parse_args()
    
    # 生成默认目录名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.test_dir is None:
        args.test_dir = f"Test_Results_{timestamp}"
    
    if args.analysis_dir is None:
        args.analysis_dir = f"Analysis_{timestamp}"
    
    print("\n" + "🎯 " * 20)
    print("RPS COMPLETE TESTING AND ANALYSIS PIPELINE")
    print("🎯 " * 20)
    
    print(f"\nConfiguration:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Seats: {args.seats}")
    print(f"  Test output: {args.test_dir}")
    print(f"  Analysis output: {args.analysis_dir}")
    
    # Step 1: 运行测试
    if not args.skip_test:
        print("\n" + "="*60)
        print("STEP 1: RUNNING MULTI-SEED TEST")
        print("="*60)
        
        # 构建test_v4命令
        test_cmd = f"python test_v4_multi_seed.py "
        test_cmd += f"--rounds {args.rounds} "
        test_cmd += f"--seeds {args.seeds} "
        test_cmd += f"--seats {args.seats} "
        test_cmd += f"--input-dir {args.input_dir} "
        test_cmd += f"--output-dir {args.test_dir} "
        test_cmd += f"--history-k {args.history_k} "
        test_cmd += f"--warmup {args.warmup} "
        
        if args.use_onehot:
            test_cmd += "--use-onehot "
        
        success = run_command(test_cmd, "Running test_v4_multi_seed.py")
        
        if not success:
            print("\n❌ Testing failed. Aborting.")
            return 1
    else:
        print("\n⏩ Skipping test phase (--skip-test)")
    
    # Step 2: 运行分析
    if not args.skip_analysis:
        print("\n" + "="*60)
        print("STEP 2: RUNNING ANALYSIS SCRIPTS")
        print("="*60)
        
        # 使用test输出目录作为分析输入
        analysis_input = args.test_dir
        
        # 2.1 运行analyze_multi_seed_1.py
        analysis1_dir = os.path.join(args.analysis_dir, "analysis_1")
        cmd1 = f"python analyze_multi_seed_1.py "
        cmd1 += f"--input-dir {analysis_input} "
        cmd1 += f"--output-dir {analysis1_dir}"
        
        success1 = run_command(cmd1, "Running analyze_multi_seed_1.py")
        
        # 2.2 运行analyze_multi_seed_2.py
        analysis2_dir = os.path.join(args.analysis_dir, "analysis_2")
        cmd2 = f"python analyze_multi_seed_2.py "
        cmd2 += f"--input-dir {analysis_input} "
        cmd2 += f"--out-dir {analysis2_dir} "
        cmd2 += f"--palette nature"  # 可选择其他调色板
        
        success2 = run_command(cmd2, "Running analyze_multi_seed_2.py")
        
        # 2.3 运行analyze_multi_seed_Lipschitz.py
        analysis3_dir = os.path.join(args.analysis_dir, "lipschitz_analysis")
        cmd3 = f"python analyze_multi_seed_Lipschitz.py "
        cmd3 += f"--input-dir {analysis_input} "
        cmd3 += f"--output-dir {analysis3_dir} "
        cmd3 += f"--palette nature"
        
        success3 = run_command(cmd3, "Running analyze_multi_seed_Lipschitz.py")
        
        # 汇总分析结果
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if success1:
            print(f"✅ Analysis 1 completed: {analysis1_dir}")
            print(f"   - Score distributions")
            print(f"   - Confidence intervals")
            print(f"   - Performance evolution")
        else:
            print(f"❌ Analysis 1 failed")
        
        if success2:
            print(f"✅ Analysis 2 completed: {analysis2_dir}")
            print(f"   - Method statistics")
            print(f"   - Win rate distributions")
            print(f"   - Statistical tests")
        else:
            print(f"❌ Analysis 2 failed")
        
        if success3:
            print(f"✅ Lipschitz analysis completed: {analysis3_dir}")
            print(f"   - Lipschitz bound verification")
            print(f"   - Distribution analysis")
            print(f"   - Agent comparisons")
        else:
            print(f"❌ Lipschitz analysis failed")
    else:
        print("\n⏩ Skipping analysis phase (--skip-analysis)")
    
    # Step 3: 生成最终报告
    print("\n" + "="*60)
    print("STEP 3: FINAL SUMMARY")
    print("="*60)
    
    print("\n📁 Output Structure:")
    print(f"""
{args.test_dir}/
├── models_seed*/           # Saved models for each seed
├── experiment_metadata.json # Experiment configuration
├── RPS_train_statistics.csv # Cross-seed statistics
├── RPS_train_all_seeds.csv  # All seed data
├── RPS_train_summary_seed*.csv # Per-seed summaries
├── RPS_record_seed*.csv     # Detailed game records
└── lipschitz_analysis/      # Lipschitz data
    └── lipschitz_seed*.csv

{args.analysis_dir}/
├── analysis_1/              # Basic analysis results
│   ├── figures/             # Visualizations
│   └── *.csv                # Statistics
├── analysis_2/              # Advanced analysis
│   ├── figures/             # More visualizations
│   └── tables/              # Statistical tables
└── lipschitz_analysis/      # Lipschitz analysis
    ├── lipschitz_figures/   # Lipschitz plots
    └── lipschitz_report.txt # Text report
""")
    
    print("\n✅ Pipeline completed successfully!")
    print(f"\nNext steps:")
    print(f"1. Check test results in: {args.test_dir}")
    print(f"2. View analysis outputs in: {args.analysis_dir}")
    print(f"3. Review Lipschitz report: {args.analysis_dir}/lipschitz_analysis/lipschitz_report.txt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())