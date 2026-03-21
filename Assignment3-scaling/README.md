# CS336 Assignment 3: Scaling Laws

> CS336 Spring 2025 · Stanford University · Assignment 3

## Assignment Description

> 本 Assignment 要求学生从零复现 Chinchilla 缩放定律的两条核心分析路径：
>
> 1. **IsoFLOPs 分析**：在固定计算预算 C 下扫描不同 (N, D) 配置，找出每个预算截面的最优配置点，再对多组最优点做幂律拟合，外推预测 10²³ 和 10²⁴ FLOPs 量级下的最优模型规模与数据规模。
> 2. **参数化损失拟合**：通过调用远程训练 API 采集 (N, D, L) 实验数据，用 L-BFGS-B 优化器拟合 Chinchilla 损失公式 `L(N,D) = E + A/N^α + B/D^β` 的五个参数，最终在拉格朗日约束 C = 6ND 下反推 10¹⁹ FLOPs 预算对应的最优模型结构。

---

## Tech Stack

| 类别 | 详情 |
|------|------|
| 语言 | Python 3.10+ |
| 数值计算 | NumPy（向量化运算、对数空间变换） |
| 科学优化 | SciPy（`curve_fit` 线性拟合、`minimize` L-BFGS-B 非线性优化） |
| 可视化 | Matplotlib（双对数坐标缩放定律外推图） |
| 数据交互 | requests（远程 HyperTuring API 训练集群通信） |
| 包管理 | uv |

---

## Core Knowledge Points

- **IsoFLOPs 分析**：固定计算预算 C，在不同参数量 N 与词元数 D 的组合上扫描验证损失，取每条 IsoFLOPs 曲线的最低点构成最优配置序列，再对该序列做幂律拟合。理论依据为 Chinchilla 论文（Hoffmann et al., 2022）提出的计算最优训练框架。

- **Chinchilla 损失公式**：`L(N, D) = E + A/N^α + B/D^β`，其中 E 为不可约损失（数据本身的熵），`A/N^α` 为容量惩罚项，`B/D^β` 为数据惩罚项。Chinchilla 原文拟合结果约为 α ≈ β ≈ 0.35，揭示了模型规模与数据规模应等比例扩展的核心结论。

- **幂律拟合**：`N_opt(C) = A · C^a`，`D_opt(C) = B · C^b`。理论上 a + b ≈ 1（由 C = 6ND 的量纲约束推导）。通过对数线性化将非线性幂律转化为线性回归问题，规避了 Levenberg-Marquardt 算法在高曲率区域的发散风险。

- **L-BFGS-B 优化**：拟牛顿法变体，通过有限内存近似海森矩阵实现二阶收敛速度，同时支持参数边界约束（Bounded）。目标函数采用对数残差平方和 `mean((log L_pred - log L_actual)²)`，等价于最小化相对误差，避免大损失值样本主导梯度。

- **对数空间线性化**：对数均匀采样（log-uniform sampling）在超参数搜索中确保每个数量级内的采样密度均等；对数空间线性回归将幂律拟合转化为标准最小二乘问题，提升数值稳定性。

- **拉格朗日约束最优化**：给定固定预算 C，通过等式约束 C = 6ND 将 D 表达为 N 的函数，把二元损失最小化问题降维为一元搜索，即在"预算超平面"上寻找损失曲面的全局最低点，这正是 Chinchilla 计算最优前沿（compute-optimal frontier）的数学本质。

---

## Code Structure

```
Assignment3-scaling/
├── cs336_scaling/
│   ├── chinchilla_isoflops_scaling.py    # IsoFLOPs 分析：分组、最优点提取、幂律拟合、外推可视化
│   └── chinchilla_scaling_laws_fitting.py # 参数化拟合：API 通信、采样网格、L-BFGS-B 拟合、拉格朗日反推
├── data/
│   └── isoflops_curves.json              # IsoFLOPs 实验数据（compute_budget / parameters / final_loss）
├── isoflops_scaling_laws_extrapolation.png  # 双子图缩放定律外推可视化结果
├── cs336_spring2025_assignment3_scaling.pdf # 原始题目说明
└── README.md
```

### chinchilla_isoflops_scaling.py 核心流程

1. 加载 JSON 数据，按 `compute_budget` 分组构建 IsoFLOPs 截面
2. 对每个截面选取 `final_loss` 最小的实验，由 `C = 6ND` 逆推 D
3. 将 (C, N_opt) 和 (C, D_opt) 映射至对数空间，用 `scipy.optimize.curve_fit` 拟合线性方程
4. 将拟合参数还原为幂律系数，外推至 10²³ 和 10²⁴ FLOPs
5. 输出双对数坐标散点图 + 外推曲线

### chinchilla_scaling_laws_fitting.py 核心流程

1. 调用 HyperTuring API 查询历史算力消耗，设置 2×10¹⁸ FLOPs 安全上限
2. `engineer_strategic_sampling_grid()` 生成跨越 10¹⁵～3×10¹⁷ 的对数均匀采样网格
3. 逐个提交训练配置，收集 (N, D, L) 观测三元组
4. L-BFGS-B 拟合 Chinchilla 五参数 (E, A, B, α, β)
5. 拉格朗日约束搜索：固定 C = 10¹⁹，在 N ∈ [10⁷, 10¹⁰] 上一维扫描最优参数量

---

## Key Results

| 分析方法 | 外推目标 | 最优参数量 N_opt | 最优数据量 D_opt |
|----------|----------|-----------------|-----------------|
| IsoFLOPs 幂律外推 | 10²³ FLOPs | 见运行输出 | 见运行输出 |
| IsoFLOPs 幂律外推 | 10²⁴ FLOPs | 见运行输出 | 见运行输出 |
| Chinchilla 损失拟合 + 拉格朗日 | 10¹⁹ FLOPs | 见运行输出 | 由 C=6ND 推导 |

幂律守恒验证：拟合得到的指数 a + b 应高度逼近 1.0，偏差越小说明数据质量越高。

---

## Setup

```sh
# 安装 uv（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加依赖
uv add numpy scipy matplotlib requests

# 运行 IsoFLOPs 分析
uv run python cs336_scaling/chinchilla_isoflops_scaling.py

# 运行参数化损失拟合（需要 Stanford HyperTuring API 访问权限）
uv run python cs336_scaling/chinchilla_scaling_laws_fitting.py
```

---

## References

- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). DeepMind.
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models*. OpenAI.
- CS336 Spring 2025 Assignment 3 Handout: `cs336_spring2025_assignment3_scaling.pdf`
