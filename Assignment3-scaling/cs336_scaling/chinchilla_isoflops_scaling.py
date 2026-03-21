"""
CS336 Assignment 3 - Problem (chinchilla_isoflops)
用于解析 IsoFLOPs 数据，并在对数空间中执行高稳定性的线性拟合，
最终将计算最优模型与数据规模严格外推至 10^24 FLOPs 级别。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict

# ─────────────────────────────────────────
# [linear_log_space_model]
# Purpose: 定义对数空间中的线性模型，用于 curve_fit 拟合幂律关系
# Key concept: 幂律 y = A * C^a 两边取对数后变为线性形式
#              log(y) = a * log(C) + log(A)，从而可以用线性最小二乘稳定求解
# ─────────────────────────────────────────
def linear_log_space_model(log_C, log_factor, exponent):
    """
    定义双边对数转换后的线性方程流形。
    公式: log(y) = exponent * log(C) + log_factor
    通过将优化空间从非线性转移至线性，彻底消除 Levenberg-Marquardt 算法发散风险。
    """
    return exponent * log_C + log_factor


# ─────────────────────────────────────────
# [canonical_power_law]
# Purpose: 将对数空间拟合得到的参数还原为原始幂律预测函数，用于外推与绘图
# Key concept: 幂律函数 y = factor * C^exponent，是描述 LLM 缩放行为的核心数学工具
# ─────────────────────────────────────────
def canonical_power_law(C, factor, exponent):
    """还原至经典幂律域中的预测函数"""
    return factor * (C ** exponent)


# ─────────────────────────────────────────
# [fit_and_extrapolate_isoflops]
# Purpose: 主函数，完成从原始 JSON 数据到缩放定律拟合与外推的全流程
# Key concept: IsoFLOPs 方法——固定计算预算 C，通过扫描不同 (N, D) 组合找到
#              最优配置，再对多个预算截面的最优点做幂律拟合，得到
#              N_opt(C) 与 D_opt(C) 的缩放关系
# ─────────────────────────────────────────
def fit_and_extrapolate_isoflops(data_path: str):

    # 从磁盘加载实验记录，每条记录包含 compute_budget、parameters、final_loss
    with open(data_path, 'r') as file:
        training_runs = json.load(file)

    # 按 FLOPs 计算预算对实验切片进行分组，以构建 IsoFLOPs 截面
    budget_cross_sections = defaultdict(list)
    for run in training_runs:
        budget = run["compute_budget"]
        budget_cross_sections[budget].append(run)

    # 收集每个预算截面的最优 (C, N, D) 三元组
    optimal_C_series = []
    optimal_N_series = []
    optimal_D_series = []

    # ─────────────────────────────────────────
    # [最优配置选取循环]
    # Purpose: 遍历每个 IsoFLOPs 截面，选出该截面内验证损失最小的实验配置
    # Key concept: IsoFLOPs 截面的核心思想是：在相同算力预算下，存在一个
    #              "最优参数量 N_opt"，既不欠拟合也不过拟合
    # ─────────────────────────────────────────
    # 定位每个预算截面内的绝对最优损失配置
    for budget, cross_section in sorted(budget_cross_sections.items()):
        # 抛弃平滑拟合，直接依据经验观察选取最低验证损失的配置
        best_run = min(cross_section, key=lambda experiment: experiment["final_loss"])

        C_i = best_run["compute_budget"]
        N_i = best_run["parameters"]
        # 根据计算约束定理 C = 6 * N * D 逆推词元规模
        # 其中 6 来自反向传播的 FLOPs 估算：前向 2ND + 反向 4ND ≈ 6ND
        D_i = C_i / (6.0 * N_i)

        optimal_C_series.append(C_i)
        optimal_N_series.append(N_i)
        optimal_D_series.append(D_i)

    # 转为 NumPy 数组以支持向量化运算
    C_array = np.array(optimal_C_series)
    N_array = np.array(optimal_N_series)
    D_array = np.array(optimal_D_series)

    # ─────────────────────────────────────────
    # [对数空间幂律拟合]
    # Purpose: 在对数空间中对 (C, N_opt) 和 (C, D_opt) 分别做线性回归，
    #          拟合得到 N_opt = A * C^a 和 D_opt = B * C^b 的幂律参数
    # Key concept: 对数线性化技巧，将非线性幂律转化为线性问题，
    #              避免非线性优化器的梯度爆炸/发散问题
    # ─────────────────────────────────────────
    # 阶段 3：映射至对数连续空间执行稳定数值优化
    log_C = np.log(C_array)
    log_N = np.log(N_array)
    log_D = np.log(D_array)

    # 针对 N(#params) 拟合：求解 log(N) = a * log(C) + log(A)
    popt_N, _ = curve_fit(linear_log_space_model, log_C, log_N)
    log_A, exponent_a = popt_N
    factor_A = np.exp(log_A)  # 将对数截距还原为幂律系数 A

    # 针对 D(#Tokens) 拟合：求解 log(D) = b * log(C) + log(B)
    popt_D, _ = curve_fit(linear_log_space_model, log_C, log_D)
    log_B, exponent_b = popt_D
    factor_B = np.exp(log_B)  # 将对数截距还原为幂律系数 B

    # ─────────────────────────────────────────
    # [外推至目标算力预算]
    # Purpose: 利用拟合得到的幂律方程，外推预测 10^23 和 10^24 FLOPs 下的最优配置
    # Key concept: 幂律外推的合理性依赖于缩放定律在更大算力区间的延续性假设
    # ─────────────────────────────────────────
    # 外推
    target_horizons = [1e23, 1e24]
    extrapolated_N = [canonical_power_law(target, factor_A, exponent_a) for target in target_horizons]
    extrapolated_D = [canonical_power_law(target, factor_B, exponent_b) for target in target_horizons]

    # 验证缩放系数的理论守恒性 (a + b 应当高度逼近 1.0)
    # 理论依据：C = 6ND => C^1 ∝ N^(1/a) * D^(1/b)，由量纲分析可得 a + b ≈ 1
    print(f"理论检验: a + b = {exponent_a + exponent_b:.4f}")

    # 生成从最小观测预算到 10^24 的连续对数刻度预算序列，用于绘制平滑外推曲线
    C_continuous = np.logspace(np.log10(min(C_array)), 24, 500)
    N_continuous = canonical_power_law(C_continuous, factor_A, exponent_a)
    D_continuous = canonical_power_law(C_continuous, factor_B, exponent_b)

    # ─────────────────────────────────────────
    # [可视化：双子图缩放定律外推图]
    # Purpose: 在双对数坐标系中同时展示经验观测点与幂律外推曲线，
    #          直观呈现 N_opt 和 D_opt 随计算预算的缩放趋势
    # Key concept: 双对数坐标（log-log plot）下幂律关系呈直线，
    #              便于直观判断拟合质量与外推合理性
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 绘制参数量外推曲线
    axes[0].scatter(C_array, N_array, color='#1f77b4', s=60, label='Empirical N_opt (Training Data)')
    axes[0].plot(C_continuous, N_continuous, color='#d62728', linestyle='dashed', linewidth=2.5,
              label=f'Extrapolation: N = {factor_A:.2e} * C^{exponent_a:.4f}')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Total Compute Budget C (FLOPs)', fontsize=12)
    axes[0].set_ylabel('Optimal Parameter Count N (Parameters)', fontsize=12)
    axes[0].set_title('Neural Scaling: Optimal Parameters vs. Compute', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, which="both", linestyle='--', alpha=0.6)

    # 绘制数据量外推曲线
    axes[1].scatter(C_array, D_array, color='#2ca02c', s=60, label='Empirical D_opt (Derived Tokens)')
    axes[1].plot(C_continuous, D_continuous, color='#ff7f0e', linestyle='dashed', linewidth=2.5,
              label=f'Extrapolation: D = {factor_B:.2e} * C^{exponent_b:.4f}')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Total Compute Budget C (FLOPs)', fontsize=12)
    axes[1].set_ylabel('Optimal Dataset Size D (Tokens)', fontsize=12)
    axes[1].set_title('Neural Scaling: Optimal Tokens vs. Compute', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, which="both", linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('isoflops_scaling_laws_extrapolation.png', dpi=300)
    print("图表已成功保存至 'isoflops_scaling_laws_extrapolation.png'")

    # 输出拟合方程和外推结果
    print("\n【缩放定律拟合方程式】")
    print(f"模型规模外推方程: N_opt(C) = {factor_A:.4e} * C^{exponent_a:.4f}")
    print(f"数据规模外推方程: D_opt(C) = {factor_B:.4e} * C^{exponent_b:.4f}")

    print("\n【针对 10^23 和 10^24 FLOPs 的单句响应】")
    print(
        f"在计算预算为 10^23 FLOPs 时，预测的最优模型规模为 {extrapolated_N[0]:.2e} 个参数；"
        f"而在计算预算为 10^24 FLOPs 时，预测的最优模型规模将达到 {extrapolated_N[1]:.2e} 个参数。")
    print(
        f"在计算预算为 10^23 FLOPs 时，预测的最优数据集规模为 {extrapolated_D[0]:.2e} 个Tokens；"
        f"而在计算预算为 10^24 FLOPs 时，预测的最优数据集规模将达到 {extrapolated_D[1]:.2e} 个Tokens。")


if __name__ == "__main__":
    fit_and_extrapolate_isoflops(r"C:\Users\liu_j\Desktop\SJTU\AI\LLM\CS336\assignment3-scaling-main\data\isoflops_curves.json")
