# RPS v3.1 Seaborn Enhanced Analysis

## 🎨 主要改进

将原始的matplotlib绘图升级为**论文出版级的Seaborn可视化**，同时保持数据处理逻辑不变。

## ✨ 新特性

### 1. 出版级视觉效果
- **专业配色方案**：支持8种预设配色
  - `nature` - Nature期刊风格
  - `science` - Science期刊风格  
  - `cell` - Cell期刊风格
  - `deep` - Seaborn深色
  - `muted` - 柔和色调
  - `bright` - 明亮色调
  - `colorblind` - 色盲友好
  - `husl` - 均匀色彩空间

### 2. 增强的可视化类型

#### 原版（仅2种图表）
- 性能演化曲线（简单线图）
- 胜率直方图（基础直方图）

#### 增强版（7种图表）
- **性能演化曲线** - 带置信区间、标注、双线（median+mean）
- **胜率直方图** - 带KDE曲线、均值/中位数标记
- **小提琴图** - 展示完整分布形状
- **箱线图** - 带数据点叠加
- **汇总仪表板** - 4合1综合分析
- **相关性热力图** - 方法间相关性
- **风险-收益散点图** - std vs mean分析



## 📊 使用方法

### 基础用法（与原版相同）
```bash
python analyze_multi_seed_v3_1_seaborn.py \
    --input-dir RPS_train_summary \
    --out-dir RPS_analysis_v3_seaborn
```

### 选择配色方案
```bash
# Nature风格（默认）
python analyze_multi_seed_v3_1_seaborn.py --palette nature

# Science风格
python analyze_multi_seed_v3_1_seaborn.py --palette science

# Cell风格
python analyze_multi_seed_v3_1_seaborn.py --palette cell
```

### 完整参数
```bash
python analyze_multi_seed_v3_1_seaborn.py \
    --input-dir RPS_train_summary \      # 输入目录
    --out-dir RPS_analysis_seaborn \     # 输出目录
    --methods "method1,method2" \         # 指定方法（可选）
    --max-steps 10000 \                  # 最大步数（可选）
    --palette nature                      # 配色方案
```

## 📈 输出文件

### 表格（与原版相同）
- `tables/scores_by_seed.csv` - 种子得分
- `tables/method_stats.csv` - 方法统计
- `tables/seat_method_stats.csv` - 席位统计
- `tables/nonparam_wilcoxon_holm.csv` - 统计检验
- `tables/winrate_distribution.csv` - 胜率分布
- `tables/method_downside.csv` - 下行风险
- `tables/perf_evolution_*.csv` - 性能演化数据

### 图像（增强版）
- `figures/perf_curve_*.png` - 性能演化曲线（增强）
- `figures/winrate_hist_*.png` - 胜率直方图（增强）
- `figures/winrate_violin_all.png` - 小提琴图（新增）
- `figures/winrate_boxplot_all.png` - 箱线图（新增）
- `figures/summary_dashboard.png` - 汇总仪表板（新增）

## 🎯 可视化对比

### 性能演化曲线
| 特性 | 原版 | Seaborn版 |
|-----|------|----------|
| 主线 | 单色细线 | 彩色粗线 |
| 置信区间 | 简单填充 | 渐变透明 |
| 均值线 | 无 | 虚线显示 |
| 零线 | 无 | 参考线 |
| 最终值标注 | 无 | 智能标注框 |

### 胜率直方图
| 特性 | 原版 | Seaborn版 |
|-----|------|----------|
| 直方图 | 基础柱状 | 带透明度 |
| KDE曲线 | 无 | 平滑密度曲线 |
| 统计线 | 无 | 均值/中位数标记 |
| 图例 | 无 | 详细图例 |

## 🔧 依赖要求

```bash
# 核心依赖（与原版相同）
numpy
pandas
tqdm
scipy

# 可视化依赖（增强）
matplotlib>=3.3.0
seaborn>=0.11.0
```

## 💡 选择建议

### 何时使用原版
- 快速分析
- 资源受限环境
- 不需要美观图表

### 何时使用Seaborn版
- **论文发表** ✓
- **技术报告** ✓
- **演示展示** ✓
- **项目文档** ✓

## 📝 注意事项

1. **兼容性**：数据处理逻辑完全相同，仅改进可视化
2. **性能**：生成图表略慢（因为300 DPI和更多细节）
3. **文件大小**：PNG文件更大（高分辨率）

## 🎨 配色方案预览

### Nature (默认)
深色专业：`#374E55` `#DF8F44` `#00A1D5` `#B24745`

### Science
明亮对比：`#0173B2` `#DE8F05` `#029E73` `#CC78BC`

### Cell
柔和学术：`#0073B7` `#E69F00` `#009E73` `#F0E442`

## 📊 示例输出质量

- **分辨率**：300 DPI（可直接用于期刊）
- **格式**：PNG with white background
- **字体**：Arial/DejaVu Sans (跨平台)
- **尺寸**：自动调整以适应内容


