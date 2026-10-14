# 自适应探索推荐算法实验代码

本项目复现 Piao 等人的 Human–AI adaptive dynamics 模型，并比较原论文基线、固定比例探索和基于近期推荐主题熵的自适应探索。项目包含代码、正式实验结果和可复现实验配置，实验报告作为单独文件提交。

## 环境

使用 Miniforge：

```powershell
conda env create -f environment.yml
conda activate adaptive_ic
python -m pip check
```

`environment.yml` 参考原项目的 Python 3.8 / NumPy 1.23.5 环境配置，并固定使用 OpenBLAS，以保证当前 Windows 环境中的矩阵运算稳定。由于 conda-forge 当前索引中已不再提供 Python 3.8.11，环境会解析可用的最新 Python 3.8 补丁版本。

如果环境已经存在，可使用：

```powershell
conda env update -n adaptive_ic -f environment.yml
conda activate adaptive_ic
```

## 数据检查

初始化数据已放在 `data/video` 和 `data/news`。运行：

```powershell
conda activate adaptive_ic
python src/data_preprocess.py
```

## 小规模测试

下列命令只运行几名用户和几轮互动，不会启动完整实验：

```powershell
conda activate adaptive_ic
python scripts/run_reproduction.py --quick
python scripts/run_comparison.py --quick
python scripts/run_sensitivity.py --quick
```

小规模测试结果会自动写入 `results/smoke`，与 `results/raw` 下的正式结果分开保存，默认绘图时不纳入小规模测试结果。

## 正式实验

```powershell
conda activate adaptive_ic
python scripts/run_reproduction.py --dataset both
python scripts/run_comparison.py --dataset both
python scripts/run_sensitivity.py
python src/plotting.py --result-root results/raw --figure-root results/figures
```

项目默认采用课程作业规模，避免直接运行论文的大规模模拟：论文复现为 150 用户、500 轮；三方法比较为 200 用户、500 轮；敏感性实验为 150 用户、500 轮。数据集、物品数、方法、参数网格和比较实验的三个随机种子保持不变。若只需要先跑单个数据集，可使用 `--dataset video` 或 `--dataset news`；如需临时调整规模，再使用 `--users` 和 `--iters` 覆盖默认值。

正式运行器每完成 100 轮会把模拟状态保存到项目内的独立断点目录，再次使用相同输出路径和配置运行时会自动从最近检查点继续。若要进行一次全新的独立运行，请指定新的 `--output-root`。

## 指标

- `mean_positive_feedback_rate`：模拟正反馈率，不是真实线上点击率。
- `mean_negative_feedback_rate`：模拟负反馈率。
- `mean_recommendation_entropy`：当前推荐主题的归一化熵。
- `mean_cumulative_entropy`：截至当前轮的累计推荐主题归一化熵。
- `relative_entropy`：推荐熵除以用户真实兴趣熵。
- `topic_coverage`：累计推荐中出现过的主题比例。
- `over_recommendation_mass`：推荐概率落在过度推荐主题上的比例。
- `under_recommendation_mass`：推荐概率落在推荐不足主题上的比例。
- `over_recommendation_degree` 和 `under_recommendation_degree`：按原论文实现，用受影响主题的真实兴趣质量归一化偏差程度；不足程度保留负号。

实验参数由 `configs/video_baseline.yaml`、`configs/news_baseline.yaml` 和 `configs/experiment_grid.yaml` 读取；命令行参数只覆盖需要临时调整的规模参数。

## 结果位置

每次实验写入指定输出目录下独立的 dataset/method/seed 目录，含配置、指标 CSV、随机种子和运行时间。默认绘图只汇总 `results/raw` 中的正式结果，并排除旧版本遗留的 5×5/8×8 smoke 记录；如确实需要查看 smoke 结果，可使用 `--include-smoke`。

## 实现说明

物品主要按单主题表示，物品主题按经验流行度独立抽样；代码按主题聚合计算相似度，再从主题物品池无放回抽取物品。这与作者 item-level softmax 在 one-hot 物品特征下等价，避免每轮对全部物品重复计算。自适应控制器只读取用户过去收到的推荐主题，不读取真实兴趣向量。

用户兴趣初始化沿用作者公开实现：读取 `user_alpha_mean_*` 后将 Dirichlet 参数整体乘以 15。这个缩放保持兴趣均值方向不变，但提高浓度、降低抽样兴趣分布的离散程度；因此它属于复现设置，报告中应明确说明。

当前实现固定使用规划中的 `alpha=0`；如果传入非零 `alpha`，程序会明确拒绝，而不会静默忽略该参数。

