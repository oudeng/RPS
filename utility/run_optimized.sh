#!/bin/bash
# run_optimized.sh - 优化后的训练运行脚本

echo "=================================================="
echo "RPS训练系统 - GPU优化运行脚本"
echo "=================================================="

# 检查参数
if [ "$#" -lt 1 ]; then
    echo "用法: ./run_optimized.sh <配置级别>"
    echo ""
    echo "配置级别:"
    echo "  1 - 测试配置（1轮，1种子，快速验证）"
    echo "  2 - 轻量配置（10轮，3种子，GPU ~30%）"
    echo "  3 - 标准配置（100轮，5种子，GPU ~40%）"
    echo "  4 - 优化配置（100轮，5种子，GPU ~60%）"
    echo "  5 - 激进配置（500轮，10种子，GPU ~70%）"
    echo "  6 - 极限配置（500轮，10种子，GPU ~80%）"
    exit 1
fi

CONFIG=$1

# 设置通用参数
SEATS="agent_seats.csv"
OUTPUT_DIR="RPS_train_summary"

case $CONFIG in
    1)
        echo "运行: 测试配置"
        echo "预期: 1分钟完成，验证环境"
        python train_v3_multi_seed.py \
            --rounds 1 \
            --seeds 42 \
            --batch-size 32 \
            --batch-freq 32 \
            --output-dir "${OUTPUT_DIR}_test"
        ;;
    
    2)
        echo "运行: 轻量配置"
        echo "预期: GPU ~30%，5分钟完成"
        python train_v3_multi_seed.py \
            --rounds 10 \
            --seeds 1,2,3 \
            --batch-size 64 \
            --batch-freq 100 \
            --output-dir "${OUTPUT_DIR}_light"
        ;;
    
    3)
        echo "运行: 标准配置"
        echo "预期: GPU ~40%，30分钟完成"
        python train_v3_multi_seed.py \
            --rounds 100 \
            --seeds 1,2,3,5,8 \
            --batch-size 64 \
            --batch-freq 64 \
            --output-dir "${OUTPUT_DIR}_standard"
        ;;
    
    4)
        echo "运行: 优化配置 ⭐推荐"
        echo "预期: GPU ~60%，30分钟完成"
        echo "命令:"
        echo "  python train_v3_multi_seed.py \\"
        echo "    --rounds 100 \\"
        echo "    --seeds 1,2,3,5,8 \\"
        echo "    --batch-size 128 \\"
        echo "    --batch-freq 128"
        
        python train_v3_multi_seed.py \
            --rounds 100 \
            --seeds 1,2,3,5,8 \
            --batch-size 128 \
            --batch-freq 128 \
            --output-dir "${OUTPUT_DIR}_optimized"
        ;;
    
    5)
        echo "运行: 激进配置"
        echo "预期: GPU ~70%，3小时完成"
        echo "警告: 请确保有足够的时间和显存"
        
        read -p "确认运行激进配置? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "已取消"
            exit 0
        fi
        
        python train_v3_multi_seed.py \
            --rounds 500 \
            --seeds 1,2,3,5,8,13,21,34,55,89 \
            --batch-size 256 \
            --batch-freq 200 \
            --output-dir "${OUTPUT_DIR}_aggressive"
        ;;
    
    6)
        echo "运行: 极限配置"
        echo "预期: GPU ~80%，3小时完成"
        echo "警告: 需要48GB显存！"
        
        read -p "确认运行极限配置? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "已取消"
            exit 0
        fi
        
        python train_v3_multi_seed.py \
            --rounds 500 \
            --seeds 1,2,3,5,8,13,21,34,55,89 \
            --batch-size 512 \
            --batch-freq 256 \
            --output-dir "${OUTPUT_DIR}_extreme"
        ;;
    
    *)
        echo "错误: 无效的配置级别"
        exit 1
        ;;
esac

echo ""
echo "训练完成！"
echo "结果保存在: ${OUTPUT_DIR}_*/"
echo ""
echo "分析结果:"
echo "  python analyze_multi_seed.py --input-dir ${OUTPUT_DIR}_*"
