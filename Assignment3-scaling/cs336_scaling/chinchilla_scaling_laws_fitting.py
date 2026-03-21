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

ABSOLUTE_BUDGET_CEILING = 2e18   # 绝对算力上限：2×10^18 FLOPs
TARGET_EXTRAPOLATION_BUDGET = 1e19  # 外推目标算力：10^19 FLOPs


# ─────────────────────────────────────────
# [compute_non_embedding_parameters]
# Purpose: 计算 Transformer 模型去除 Embedding 层后的参数量，
#          用于缩放定律分析中"有效参数量 N"的标准化度量
# Key concept: 12 * L * d² 公式来源——标准 Transformer 每层包含：
#   - 注意力模块：Q、K、V、O 四个投影矩阵，每个形状为 (d, d)，共 4d²
#   - FFN 模块：两个线性层，通常扩展比为 4，即 d→4d→d，共 2×(d×4d) = 8d²
#   - 合计每层：4d² + 8d² = 12d²，L 层共 12Ld²
#   - Embedding 与 LM Head 的参数量受词表大小影响，在缩放分析中通常予以排除
# ─────────────────────────────────────────
def compute_non_embedding_parameters(d_model: int, num_layers: int) -> int:
    """4个注意力矩阵+2层FFN"""
    return 12 * num_layers * (d_model ** 2)


# ─────────────────────────────────────────
# [derive_training_tokens]
# Purpose: 根据计算预算 C 和参数量 N 反推训练词元数 D
# Key concept: Kaplan 等人提出的计算约束公式 C ≈ 6ND
#   其中 6 = 2（前向）+ 4（反向，梯度+激活梯度各约 2 倍），
#   该估算忽略了注意力的二次项，在 d_model >> seq_len 时精度较高
# ─────────────────────────────────────────
def derive_training_tokens(train_flops: float, parameter_count: float) -> float:
    """C = 6ND"""
    return train_flops / (6.0 * parameter_count)


# ─────────────────────────────────────────
# [HyperTuringAPIClient]
# Purpose: 封装与远程 Stanford 训练集群的 HTTP 通信，
#          提供预算查询和损失测量两类接口
# Key concept: 通过远程 API 模拟实际训练实验，将"训练一个模型"
#              抽象为一次 HTTP 请求，便于在有限预算下大规模探索超参空间
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# [engineer_strategic_sampling_grid]
# Purpose: 生成覆盖多个算力量级的异构超参数探测网格，
#          在不突破 2×10^18 FLOPs 总预算上限的前提下，
#          最大化对损失曲面的信息覆盖
# Key concept: 对数均匀采样（log-uniform sampling）的必要性——
#   缩放定律关注的是跨越多个数量级的行为（如 N 从 10^6 到 10^10），
#   线性均匀采样会导致小规模区间严重欠采、大规模区间过度集中；
#   而对数均匀采样在每个数量级内保持相同的采样密度，
#   确保拟合结果不被某一量级的数据点统治，从而得到更鲁棒的幂律估计。
#   学习率同样采用对数均匀抽样，因为其对模型收敛的影响也是对数尺度上的
# ─────────────────────────────────────────
def engineer_strategic_sampling_grid() -> List:
    """
    规避全量随机扫描，生成基于对数递减规律的异构超参数探测阵列。
    该阵列的累计消耗在设计之初即受到极限测算，不会突破 2e18 的红线警戒区。
    """
    sampling_mesh = []

    # 构建异构截面的节点映射图 (FLOPs_Budget, Number_of_Samples)
    # 越大的预算截面采样越稀疏，因为单次实验代价更高
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
            # 10^uniform(-4, -3) 在对数尺度上均匀覆盖整个区间
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


# ─────────────────────────────────────────
# [parametric_loss_hypothesis]
# Purpose: 实例化 Chinchilla (Hoffmann et al., 2022) 提出的参数化损失函数，
#          用于拟合实验观测到的 (N, D, L) 三元组数据
# Key concept: Chinchilla 损失公式 L(N, D) = E + A/N^α + B/D^β
#   - E：不可约损失（irreducible loss），即数据分布固有熵，与模型大小/数据量无关
#   - A/N^α：容量惩罚项，随参数量 N 增加而减小；α 描述参数效率的衰减速率
#   - B/D^β：数据惩罚项，随训练词元数 D 增加而减小；β 描述数据效率的衰减速率
#   - Chinchilla 论文拟合结果约为 α ≈ β ≈ 0.35，A ≈ 406, B ≈ 410, E ≈ 1.69
#   该公式在双对数空间中将损失分解为三个独立的贡献项，
#   是理解"计算最优训练"（compute-optimal training）的核心数学基础
# ─────────────────────────────────────────
def parametric_loss_hypothesis(params, N_space, D_space):
    """
    实例化 Chinchilla 定理所提出的参数化经验分布公式。
    添加微小 epsilon 扰动防止除零异常及对数运算非法。
    """
    E, A, B, alpha, beta = params
    epsilon = 1e-12
    # A/N^alpha：容量不足带来的损失惩罚，N 越大惩罚越小
    penalty_capacity = A / (N_space ** alpha + epsilon)
    # B/D^beta：数据不足带来的损失惩罚，D 越大惩罚越小
    penalty_data = B / (D_space ** beta + epsilon)
    return E + penalty_capacity + penalty_data


# ─────────────────────────────────────────
# [l_bfgs_b_objective_function]
# Purpose: 定义 L-BFGS-B 优化器所最小化的目标函数（损失曲面拟合误差）
# Key concept: 使用 log 差的平方（而非直接 MSE）的原因——
#   1. 损失值 L 本身跨越多个数量级（从约 1.5 到 10+），
#      直接用 MSE 会让大损失值的样本主导梯度，忽视低损失区域的精度；
#   2. log 变换将乘性误差转化为加性误差，使得各量级的拟合精度均等，
#      等价于最小化相对误差（relative error）而非绝对误差；
#   3. Chinchilla 论文原文也采用对数空间的残差平方和作为拟合目标，
#      保持一致性有助于复现原始结果
# ─────────────────────────────────────────
def l_bfgs_b_objective_function(params, N_actual, D_actual, L_actual):
    """构建对数差异惩罚器，抵御极端发散值的异常梯度绑架"""
    L_predicted = parametric_loss_hypothesis(params, N_actual, D_actual)
    # log(L_pred) - log(L_actual) = log(L_pred / L_actual)，即对数相对误差
    log_diff = np.log(L_predicted) - np.log(L_actual)
    return np.mean(log_diff ** 2)


# ─────────────────────────────────────────
# [orchestrate_scaling_laws_pipeline]
# Purpose: 主调度函数，串联 API 通信、超参数网格扫描、L-BFGS-B 曲面拟合
#          以及拉格朗日约束最优结构反演的完整流水线
# Key concept: 整体流程遵循"实验设计 → 数据收集 → 参数估计 → 最优预测"
#              的经典统计推断框架
# ─────────────────────────────────────────
def orchestrate_scaling_laws_pipeline():
    """主调度函数：执行云端通信、数据收集与数学反推闭环"""
    agent = HyperTuringAPIClient(API_AUTHENTICATION_KEY)

    historical_burn = agent.probe_cumulative_expenditure()
    print(f"初始审计侦测: 历史已用算力为 {historical_burn:.2e} FLOPs。")

    tactical_grid = engineer_strategic_sampling_grid()
    N_vector, D_vector, L_vector = [], [], []

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

    # ─────────────────────────────────────────
    # [L-BFGS-B 非线性优化]
    # Purpose: 使用拟牛顿法 L-BFGS-B 在带边界约束的参数空间中，
    #          最小化对数残差目标函数，求解 Chinchilla 损失公式的五个参数
    # Key concept: L-BFGS-B（Limited-memory BFGS with Bounds）
    #   - 通过有限内存近似海森矩阵（Hessian），兼顾牛顿法的二阶收敛速度
    #     与梯度下降的内存效率，适合中等规模的非线性参数估计问题
    #   - bounds 约束保证参数的物理意义（如 E > 0, α ∈ (0,1)），
    #     防止优化器陷入无物理意义的局部极值
    # ─────────────────────────────────────────
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

    # ─────────────────────────────────────────
    # [lagrangian_constrained_loss（内嵌函数）]
    # Purpose: 在固定计算预算 C = 10^19 FLOPs 的约束下，
    #          将二元损失函数 L(N, D) 化简为关于 N 的一元函数
    # Key concept: 拉格朗日约束的含义——
    #   给定计算预算 C，N 与 D 通过等式约束 C = 6ND 相互绑定，
    #   即 D = C / (6N)，代入损失公式后自由度从 (N, D) 降为 N 单变量；
    #   这等价于在预算超平面上寻找损失曲面的最低点，
    #   即 argmin_N L(N, C/(6N))，是拉格朗日约束优化在解析简化后的形式。
    #   Chinchilla 论文中称此为"compute-optimal frontier"（计算最优前沿）
    # ─────────────────────────────────────────
    def lagrangian_constrained_loss(N_candidate):
        # 由预算约束 C = 6ND 推导 D，将二元问题降维为一元搜索
        D_derived = TARGET_EXTRAPOLATION_BUDGET / (6.0 * N_candidate)
        return parametric_loss_hypothesis(optimization_result.x, N_candidate, D_derived)

    # 构建高精度的一维连续对数参数空间以搜索谷底解析解
    # logspace(7, 10) 对应 N 从 10^7 到 10^10，覆盖典型 LLM 参数规模范围
    N_search_space = np.logspace(7, 10, 50000)
    L_projection_space = [lagrangian_constrained_loss(n) for n in N_search_space]

    global_min_index = np.argmin(L_projection_space)
    optimal_continuous_N = N_search_space[global_min_index]
    predicted_minimum_loss = L_projection_space[global_min_index]

    # 根据常数建立过度训练比校验
    optimal_D = TARGET_EXTRAPOLATION_BUDGET / (6 * optimal_continuous_N)
    otr_ratio = optimal_D / optimal_continuous_N  # 过度训练比（D/N），Chinchilla 结论约为 ~20

    print(f"\n===========================================================")
    print(f"面向目标预算 {TARGET_EXTRAPOLATION_BUDGET:.1e} FLOPs 的预测")
    print(f"===========================================================")
    print(f"理论绝对最优参数量 N: {optimal_continuous_N:.2e} (约 {optimal_continuous_N / 1e6:.1f} 百万参数)")
    print(f"理论极限最低验证损失 Loss: {predicted_minimum_loss:.5f}")
    print(f"内禀特征比对 (过度训练比 D/N): {otr_ratio:.2f}")


if __name__ == "__main__":
    orchestrate_scaling_laws_pipeline()
