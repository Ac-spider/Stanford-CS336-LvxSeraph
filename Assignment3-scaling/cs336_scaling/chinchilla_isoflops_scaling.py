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

def linear_log_space_model(log_C, log_factor, exponent):
    """
    定义双边对数转换后的线性方程流形。
    公式: log(y) = exponent * log(C) + log_factor
    通过将优化空间从非线性转移至线性，彻底消除 Levenberg-Marquardt 算法发散风险。
    """
    return exponent * log_C + log_factor


def canonical_power_law(C, factor, exponent):
    """还原至经典幂律域中的预测函数"""
    return factor * (C ** exponent)


def fit_and_extrapolate_isoflops(data_path: str):

    with open(data_path, 'r') as file:
        training_runs = json.load(file)

    # 按 FLOPs 计算预算对实验切片进行分组，以构建 IsoFLOPs 截面
    budget_cross_sections = defaultdict(list)
    for run in training_runs:
        budget = run["compute_budget"]
        budget_cross_sections[budget].append(run)

    optimal_C_series = []
    optimal_N_series = []
    optimal_D_series = []

    # 定位每个预算截面内的绝对最优损失配置
    for budget, cross_section in sorted(budget_cross_sections.items()):
        # 抛弃平滑拟合，直接依据经验观察选取最低验证损失的配置
        best_run = min(cross_section, key=lambda experiment: experiment["final_loss"])

        C_i = best_run["compute_budget"]
        N_i = best_run["parameters"]
        # 根据计算约束定理 C = 6 * N * D 逆推词元规模
        D_i = C_i / (6.0 * N_i)

        optimal_C_series.append(C_i)
        optimal_N_series.append(N_i)
        optimal_D_series.append(D_i)

    C_array = np.array(optimal_C_series)
    N_array = np.array(optimal_N_series)
    D_array = np.array(optimal_D_series)

    # 阶段 3：映射至对数连续空间执行稳定数值优化
    log_C = np.log(C_array)
    log_N = np.log(N_array)
    log_D = np.log(D_array)

    # 针对 N(#params) 拟合
    popt_N, _ = curve_fit(linear_log_space_model, log_C, log_N)
    log_A, exponent_a = popt_N
    factor_A = np.exp(log_A)

    # 针对 D(#Tokens) 拟合
    popt_D, _ = curve_fit(linear_log_space_model, log_C, log_D)
    log_B, exponent_b = popt_D
    factor_B = np.exp(log_B)

    # 外推
    target_horizons = [1e23, 1e24]
    extrapolated_N = [canonical_power_law(target, factor_A, exponent_a) for target in target_horizons]
    extrapolated_D = [canonical_power_law(target, factor_B, exponent_b) for target in target_horizons]

    # 验证缩放系数的理论守恒性 (a + b 应当高度逼近 1.0)
    print(f"理论检验: a + b = {exponent_a + exponent_b:.4f}")

    # 生成预算曲线
    C_continuous = np.logspace(np.log10(min(C_array)), 24, 500)
    N_continuous = canonical_power_law(C_continuous, factor_A, exponent_a)
    D_continuous = canonical_power_law(C_continuous, factor_B, exponent_b)

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
