"""
CS336 Assignment 3 - Problem (scaling_laws)
集成了自适应 API 预算监控机制、非均匀异构空间扫描、
L-BFGS-B 非线性损失曲面拟合以及基于 1e19 FLOPs 拉格朗日约束的最优结构反演。
"""

import math
import time
import requests
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple

# API 配置全局常量
API_BASE_ENDPOINT = "http://hyperturing.stanford.edu:8000"
API_AUTHENTICATION_KEY = "1234567"

ABSOLUTE_BUDGET_CEILING = 2e18
TARGET_EXTRAPOLATION_BUDGET = 1e19


def compute_non_embedding_parameters(d_model: int, num_layers: int) -> int:
    """4个注意力矩阵+2层FFN"""
    return 12 * num_layers * (d_model ** 2)


def derive_training_tokens(train_flops: float, parameter_count: float) -> float:
    """C = 6ND"""
    return train_flops / (6.0 * parameter_count)


class HyperTuringAPIClient:
    """封装对Stanford训练集群的高容错 HTTP 接口层"""

    def __init__(self, key: str):
        self.key = key

    def probe_cumulative_expenditure(self) -> float:
        """从审计端点读取系统认定的历史算力累积值，实施安全拦截阀门"""
        endpoint = f"{API_BASE_ENDPOINT}/total_flops_used"
        try:
            response = requests.get(endpoint, params={"api_key": self.key}, timeout=10)
            if response.status_code == 200:
                return float(response.json())
        except Exception as e:
            print(f"API 网络握手异常: {e}")
        return 0.0

    def query_training_surface(self, config: Dict) -> float:
        """向远程集群投递具体的模型拓扑和训练规划，并捕获最终模型收敛损失"""
        endpoint = f"{API_BASE_ENDPOINT}/loss"
        payload = config.copy()
        payload["api_key"] = self.key
        try:
            response = requests.get(endpoint, params=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("loss", None)
            else:
                print(f"服务器拒载响应 ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"损失查询端点熔断: {e}")
        return None


def engineer_strategic_sampling_grid() -> List:
    """
    规避全量随机扫描，生成基于对数递减规律的异构超参数探测阵列。
    该阵列的累计消耗在设计之初即受到极限测算，不会突破 2e18 的红线警戒区。
    """
    sampling_mesh = []

    # 构建异构截面的节点映射图 (FLOPs_Budget, Number_of_Samples)
    budget_stratification_plan = [
        (1e15, 15),
        (3e15, 15),
        (1e16, 15),
        (3e16, 15),
        (1e17, 8),
        (3e17, 1)
    ]

    # 为了保证梯度传播机制的数值稳定性，学习率将以对数尺度在限定域内抽取
    for compute_strata, sample_density in budget_stratification_plan:
        for _ in range(sample_density):
            # 将模型的层深边界映射为离散区间，并受制于 2 至 24 的 API 规则
            layers = int(np.random.randint(2, 25))
            # d_model 选择常见的 2 的幂次与标准切分点
            d_model = int(np.random.choice([64, 128, 256, 512, 1024]))

            # 多头注意力头数通常受制于 d_model / 64 的最佳工程实践
            head_dim = 64
            heads = max(2, d_model // head_dim)
            if heads > 16: heads = 16  # API 上限阻断

            # 对数均匀抽样抽取学习率，落在 [1e-4, 1e-3] 内
            lr = float(10 ** np.random.uniform(-4, -3))

            # 模型可行性防御机制：若选出的巨型架构在给定的微小算力下连一个 batch 都跑不完，则跳过
            N = compute_non_embedding_parameters(d_model, layers)
            token_exposure = compute_strata / (6 * N)
            if token_exposure < 100:
                continue

            sampling_mesh.append({
                "d_model": d_model,
                "num_layers": layers,
                "num_heads": heads,
                "batch_size": 256,  # 强制最大批次规模以榨取显存和计算吞吐峰值
                "learning_rate": lr,
                "train_flops": int(compute_strata)
            })

    return sampling_mesh


def parametric_loss_hypothesis(params, N_space, D_space):
    """
    实例化 Chinchilla 定理所提出的参数化经验分布公式。
    添加微小 epsilon 扰动防止除零异常及对数运算非法。
    """
    E, A, B, alpha, beta = params
    epsilon = 1e-12
    penalty_capacity = A / (N_space ** alpha + epsilon)
    penalty_data = B / (D_space ** beta + epsilon)
    return E + penalty_capacity + penalty_data


def l_bfgs_b_objective_function(params, N_actual, D_actual, L_actual):
    """构建对数差异惩罚器，抵御极端发散值的异常梯度绑架"""
    L_predicted = parametric_loss_hypothesis(params, N_actual, D_actual)
    log_diff = np.log(L_predicted) - np.log(L_actual)
    return np.mean(log_diff ** 2)


def orchestrate_scaling_laws_pipeline():
    """主调度函数：执行云端通信、数据收集与数学反推闭环"""
    agent = HyperTuringAPIClient(API_AUTHENTICATION_KEY)

    historical_burn = agent.probe_cumulative_expenditure()
    print(f"初始审计侦测: 历史已用算力为 {historical_burn:.2e} FLOPs。")

    tactical_grid = engineer_strategic_sampling_grid()
    N_vector, D_vector, L_vector =[],[],[]

    # 执行探测矩阵
    for idx, configuration in enumerate(tactical_grid):
        cost = configuration["train_flops"]
        projected_total = historical_burn + cost

        # 拦截器设置在理论极限的 95%
        if projected_total > (ABSOLUTE_BUDGET_CEILING * 0.95):
            print(f"\n[警报] 预测算力 {projected_total:.2e} 已抵近MAX。系统实施自动截断机制终止网格探测。")
            break

        print(f"正提交第 {idx + 1} 个训练任务，算力消耗: {cost:.2e} FLOPs...")
        empirical_loss = agent.query_training_surface(configuration)

        if empirical_loss is not None:
            N = compute_non_embedding_parameters(configuration["d_model"], configuration["num_layers"])
            D = derive_training_tokens(cost, N)

            N_vector.append(N)
            D_vector.append(D)
            L_vector.append(empirical_loss)

            historical_burn += cost
            time.sleep(0.05)

    if not N_vector:
        print("致命错误：未从服务器回收任何有效的经验测量点，无法继续推演曲面。")
        return

    # 将回收的观测值转化为张量以支持大规模向量化偏导计算
    N_tensor = np.array(N_vector)
    D_tensor = np.array(D_vector)
    L_tensor = np.array(L_vector)

    # 建立多维非线性特征空间约束
    initial_guess_vector = [1.5, 450.0, 2100.0, 0.35, 0.35]
    strict_bounds = [
        (0.1, 5.0),  # E：不可约损失，理论上在 (0, ~5) 之间
        (1.0, 10000.0),  # A：容量惩罚系数，正数且可以很大
        (1.0, 10000.0),  # B：数据惩罚系数，同上
        (0.01, 1.0),  # alpha：容量衰减指数，Chinchilla 约 0.35
        (0.01, 1.0),  # beta：数据衰减指数，Chinchilla 约 0.35
    ]

    print("\n[系统通知] 探测完成。正在启动 L-BFGS-B 非线性海森矩阵逼近求解引擎...")
    optimization_result = minimize(
        l_bfgs_b_objective_function,
        initial_guess_vector,
        args=(N_tensor, D_tensor, L_tensor),
        method='L-BFGS-B',
        bounds=strict_bounds
    )

    E_opt, A_opt, B_opt, alpha_opt, beta_opt = optimization_result.x
    print(f"\n[数学解析完成] 求解出全局最优拓扑参数坐标系:")
    print(f"E (不可约熵) = {E_opt:.4f}")
    print(f"A (容量增益) = {A_opt:.2f}, alpha = {alpha_opt:.4f}")
    print(f"B (数据增益) = {B_opt:.2f}, beta  = {beta_opt:.4f}")

    # =======================================================
    # 执行针对于 10^19 巨无霸计算阈值的超强拉格朗日逆推
    # =======================================================
    def lagrangian_constrained_loss(N_candidate):
        D_derived = TARGET_EXTRAPOLATION_BUDGET / (6.0 * N_candidate)
        return parametric_loss_hypothesis(optimization_result.x, N_candidate, D_derived)

    # 构建高精度的一维连续对数参数空间以搜索谷底解析解
    N_search_space = np.logspace(7, 10, 50000)
    L_projection_space = [lagrangian_constrained_loss(n) for n in N_search_space]

    global_min_index = np.argmin(L_projection_space)
    optimal_continuous_N = N_search_space[global_min_index]
    predicted_minimum_loss = L_projection_space[global_min_index]

    # 根据常数建立过度训练比校验
    optimal_D = TARGET_EXTRAPOLATION_BUDGET / (6 * optimal_continuous_N)
    otr_ratio = optimal_D / optimal_continuous_N

    print(f"\n===========================================================")
    print(f"面向目标预算 {TARGET_EXTRAPOLATION_BUDGET:.1e} FLOPs 的预测")
    print(f"===========================================================")
    print(f"理论绝对最优参数量 N: {optimal_continuous_N:.2e} (约 {optimal_continuous_N / 1e6:.1f} 百万参数)")
    print(f"理论极限最低验证损失 Loss: {predicted_minimum_loss:.5f}")
    print(f"内禀特征比对 (过度训练比 D/N): {otr_ratio:.2f}")


if __name__ == "__main__":
    orchestrate_scaling_laws_pipeline()
