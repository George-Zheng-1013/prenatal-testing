import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 创建results目录
if not os.path.exists("results"):
    os.makedirs("results")
    print("创建results目录")


class NIPTDataProcessor:
    """NIPT数据预处理和工具类"""

    def __init__(self):
        self.male_data = None
        self.female_data = None

    def load_data(self, file_path):
        """加载CSV数据"""
        try:
            # 读取预处理后的男胎数据（CSV格式）
            self.male_data = pd.read_csv(file_path)
            self.female_data = None  # 问题3只需要男胎数据

            # 创建列名映射：中文列名到字母列名
            column_mapping = {
                "序号": "A",
                "孕妇代码": "B",
                "年龄": "C",
                "身高": "D",
                "体重": "E",
                "末次月经": "F",
                "IVF妊娠": "G",
                "检测日期": "H",
                "检测抽血次数": "I",
                "检测孕周": "J",
                "孕妇BMI": "K",
                "原始读段数": "L",
                "在参考基因组上比对的比例": "M",
                "重复读段的比例": "N",
                "唯一比对的读段数  ": "O",  # 注意这里有多余空格
                "GC含量": "P",
                "13号染色体的Z值": "Q",
                "18号染色体的Z值": "R",
                "21号染色体的Z值": "S",
                "X染色体的Z值": "T",
                "Y染色体的Z值": "U",
                "Y染色体浓度": "V",
                "X染色体浓度": "W",
                "13号染色体的GC含量": "X",
                "18号染色体的GC含量": "Y",
                "21号染色体的GC含量": "Z",
                "被过滤掉读段数的比例": "AA",
                "染色体的非整倍体": "AB",
                "怀孕次数": "AC",
                "生产次数": "AD",
                "胎儿是否健康": "AE",
            }

            # 重命名列为字母格式以保持代码兼容性
            self.male_data = self.male_data.rename(columns=column_mapping)

            # 添加解析后的孕周列
            self.male_data["J_week"] = self.male_data["J"]  # 孕周已经预处理过

            print(f"成功加载数据: 男胎{len(self.male_data)}条")
            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def preprocess_male_data(self):
        """预处理男胎数据 - 添加数据过滤"""
        if self.male_data is None:
            print("请先加载数据")
            return None

        df = self.male_data.copy()
        original_len = len(df)

        print(f"原始数据: {original_len}条记录")

        # 1. 过滤检测孕周：保留 [10, 24) 区间
        df = df[(df["J"] >= 10) & (df["J"] < 24)]
        print(f"孕周过滤 [10w, 24w): {original_len} -> {len(df)}条记录")

        # 2. 过滤BMI极端异常值
        # 使用IQR方法识别异常值
        Q1 = df["K"].quantile(0.25)
        Q3 = df["K"].quantile(0.75)
        IQR = Q3 - Q1

        # 定义异常值范围（1.2倍IQR规则，但设置合理的BMI范围）
        lower_bound = max(Q1 - 1.2 * IQR, 15.0)  # BMI不低于15
        upper_bound = min(Q3 + 1.2 * IQR, 50.0)  # BMI不高于50

        before_bmi_filter = len(df)
        df = df[(df["K"] >= lower_bound) & (df["K"] <= upper_bound)]
        print(
            f"BMI异常值过滤 [{lower_bound:.1f}, {upper_bound:.1f}]: {before_bmi_filter} -> {len(df)}条记录"
        )

        # 3. 过滤其他必要的缺失值
        df = df.dropna(subset=["B", "K", "J", "V"])  # 孕妇代码、BMI、孕周、Y染色体浓度
        print(f"缺失值过滤: {len(df)}条记录")

        print(f"最终数据: {len(df)}条记录 (过滤掉{original_len - len(df)}条)")
        return df.reset_index(drop=True)


class Problem3Solver:
    """问题3：多因素BMI分组与达标比例约束分析"""

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.prob_model = None
        self.optimal_groups = None
        self.scaler = None

    def prepare_multifactor_features(self, df):
        """准备多因素特征"""
        features = pd.DataFrame()

        # 基础特征
        features["J_week"] = df["J_week"]
        features["K_BMI"] = df["K"]
        features["C_age"] = df["C"]
        features["D_height"] = df["D"]
        features["E_weight"] = df["E"]

        # 质量特征
        features["L_reads"] = np.log10(df["L"])
        features["M_mapped"] = df["M"]
        features["P_GC"] = df["P"]
        features["AA_filtered"] = df["AA"]

        # 派生特征
        features["BMI_age"] = features["K_BMI"] * features["C_age"]
        features["height_weight_ratio"] = features["D_height"] / features["E_weight"]
        features["BMI_week"] = features["K_BMI"] * features["J_week"]

        # 标准化身高体重为z-score
        features["height_zscore"] = (
            features["D_height"] - features["D_height"].mean()
        ) / features["D_height"].std()
        features["weight_zscore"] = (
            features["E_weight"] - features["E_weight"].mean()
        ) / features["E_weight"].std()

        return features

    def fit_probability_model(self, df, threshold=0.04):
        """拟合多因素达标概率模型"""
        print("拟合多因素达标概率模型...")

        # 准备特征和目标
        features = self.prepare_multifactor_features(df)
        target = (df["V"] >= threshold).astype(int)

        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # 使用随机森林建模概率
        self.prob_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        )
        self.prob_model.fit(X_scaled, target)

        # 评估性能
        cv_scores = cross_val_score(
            self.prob_model, X_scaled, target, cv=3, scoring="roc_auc"
        )
        print(f"达标概率模型AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        self.feature_names = features.columns.tolist()
        return self.prob_model

    def predict_attainment_probability(self, df, test_week):
        """预测个体在指定孕周的达标概率"""
        if self.prob_model is None:
            raise ValueError("概率模型尚未拟合")

        # 准备特征，设置孕周为test_week
        features = self.prepare_multifactor_features(df)
        features["J_week"] = test_week

        # 重新计算包含孕周的派生特征
        features["BMI_week"] = features["K_BMI"] * test_week

        # 标准化并预测
        X_scaled = self.scaler.transform(features)
        probabilities = self.prob_model.predict_proba(X_scaled)[:, 1]

        return probabilities

    def calculate_group_attainment_rate(self, group_df, test_week):
        """计算群体在指定孕周的达标比例。若存在组级事件时间模型，则用 F_g(t)；否则用横断面概率均值。"""
        # 若 group_df 包含 cluster 标签并且模型已拟合，使用事件时间分布
        if (
            "cluster" in group_df.columns
            and hasattr(self, "group_surv")
            and self.group_surv
        ):
            cluster_vals = group_df["cluster"].unique()
            # 若 group_df 来自单个cluster，则直接使用
            if len(cluster_vals) == 1:
                g = int(cluster_vals[0])
                res = self.group_surv.get(g, None)
                if res is not None:
                    Sg_t = self.eval_step_S(
                        np.array([test_week]), res["support"], res["S"]
                    )
                    Fg_t = 1.0 - Sg_t[0]
                    # 若估计失败则回退到横断面概率预测
                    if np.isfinite(Fg_t):
                        return float(Fg_t)
        # 回退：使用当前横断面概率平均（不改变外部接口）
        individual_probs = self.predict_attainment_probability(group_df, test_week)
        group_rate = individual_probs.mean()
        return group_rate

    def calculate_multifactor_risk(
        self,
        group_df,
        test_week,
        risk_weights=None,
        min_attain_rate=0.9,
        measurement_error=0.05,
    ):
        """计算考虑多因素和测量误差的期望风险"""
        if risk_weights is None:
            risk_weights = {
                "early": 0.8,
                "mid": 2.5,
                "late": 15.0,
                "retest": 1.5,
                "low_attain": 80.0,
                "error": 3.0,
            }

        # 计算群体达标率
        group_attain_rate = self.calculate_group_attainment_rate(group_df, test_week)

        # 考虑测量误差的影响
        # 假设测量误差导致达标率的不确定性
        effective_attain_rate = group_attain_rate * (1 - measurement_error)

        # 基础时间风险
        if test_week <= 12:
            time_risk = risk_weights["early"]
        elif test_week <= 27:
            time_risk = risk_weights["mid"]
        else:
            time_risk = risk_weights["late"]

        # 失败风险
        failure_risk = 1 - effective_attain_rate

        # 群体达标率不足的惩罚
        attain_penalty = 0
        if group_attain_rate < min_attain_rate:
            attain_penalty = risk_weights["low_attain"] * (
                min_attain_rate - group_attain_rate
            )

        # 测量误差带来的额外风险
        error_risk = risk_weights["error"] * measurement_error * failure_risk

        # 总风险
        total_risk = (
            time_risk * failure_risk
            + risk_weights["retest"] * failure_risk
            + attain_penalty
            + error_risk
        )

        return total_risk, group_attain_rate, effective_attain_rate

    def two_stage_search(
        self,
        group_df,
        pi_min=0.90,
        metric="ET",
        step0=0.25,
        step1=0.25,
        late_cap=24,
        tau=25,
        cw=None,
    ):
        """两阶段最佳时点搜索"""
        if cw is None:
            cw = {
                "early": 0.8,
                "mid": 2.5,
                "late": 15,
                "retest": 1.5,
                "short": 80,
                "err": 3,
            }

        # 1) 预计算 F_g(t)
        T0 = np.arange(10, 20 + 1e-9, step0)
        T1 = np.arange(10.5, 25 + 1e-9, step1)

        # 计算所有需要的时间点的达标率
        F = {}
        for t in np.union1d(T0, T1):
            F[t] = self.calculate_group_attainment_rate(group_df, t)

        best = None
        valid_solutions = 0  # 统计有效解的数量

        for t0 in T0:
            for t1 in T1[T1 > t0]:
                if late_cap is not None and t1 > late_cap:  # 可选硬限制
                    continue
                if F[t1] < pi_min:  # 覆盖硬约束
                    continue

                valid_solutions += 1  # 找到有效解

                if metric == "ET":
                    # 期望完成时间
                    val = t0 * F[t0] + t1 * (F[t1] - F[t0]) + tau * (1 - F[t1])
                else:
                    # 期望风险/成本
                    c0 = (
                        cw["early"]
                        if t0 <= 12
                        else (cw["mid"] if t0 <= 27 else cw["late"])
                    )
                    c1 = (
                        cw["early"]
                        if t1 <= 12
                        else (cw["mid"] if t1 <= 27 else cw["late"])
                    )
                    val = (
                        c0 * (1 - F[t0])
                        + c1 * (F[t1] - F[t0])
                        + cw["retest"] * (1 - F[t0])
                        + cw["short"] * max(0.0, pi_min - F[t1])
                        + (cw["late"] if t1 >= 28 else 0.0)
                    )

                if (best is None) or (val < best["val"]):
                    best = {
                        "t0": t0,
                        "t1": t1,
                        "val": val,
                        "F0": F[t0],
                        "F1": F[t1],
                        "retest_rate": 1 - F[t0],
                        "metric": metric,
                    }

        # 如果没有找到满足硬约束的解，选择F[t1]最高的组合
        if best is None and valid_solutions == 0:
            max_F1 = 0
            for t0 in T0:
                for t1 in T1[T1 > t0]:
                    if late_cap is not None and t1 > late_cap:
                        continue
                    if F[t1] > max_F1:
                        max_F1 = F[t1]

                        if metric == "ET":
                            val = t0 * F[t0] + t1 * (F[t1] - F[t0]) + tau * (1 - F[t1])
                        else:
                            c0 = (
                                cw["early"]
                                if t0 <= 12
                                else (cw["mid"] if t0 <= 27 else cw["late"])
                            )
                            c1 = (
                                cw["early"]
                                if t1 <= 12
                                else (cw["mid"] if t1 <= 27 else cw["late"])
                            )
                            val = (
                                c0 * (1 - F[t0])
                                + c1 * (F[t1] - F[t0])
                                + cw["retest"] * (1 - F[t0])
                                + cw["short"] * max(0.0, pi_min - F[t1])
                                + (cw["late"] if t1 >= 28 else 0.0)
                            )

                        best = {
                            "t0": t0,
                            "t1": t1,
                            "val": val,
                            "F0": F[t0],
                            "F1": F[t1],
                            "retest_rate": 1 - F[t0],
                            "metric": metric,
                        }

        return best

    def optimize_group_testing_time(
        self, group_df, time_range=(10, 25), min_attain_rate=0.9, measurement_error=0.05
    ):
        """优化单个组的最佳检测时间 - 支持两阶段"""
        # 先进行单阶段优化（保持原有逻辑）
        test_times = np.linspace(time_range[0], time_range[1], 20)

        best_time = 15  # 默认值
        best_risk = np.inf
        best_attain = 0
        best_effective_attain = 0

        for t in test_times:
            risk, attain, eff_attain = self.calculate_multifactor_risk(
                group_df,
                t,
                min_attain_rate=min_attain_rate,
                measurement_error=measurement_error,
            )

            # 优先选择满足达标率要求的时间点
            if attain >= min_attain_rate and risk < best_risk:
                best_time = t
                best_risk = risk
                best_attain = attain
                best_effective_attain = eff_attain
            elif best_risk == np.inf and attain > best_attain:
                # 如果没有满足要求的时间，选择达标率最高的
                best_time = t
                best_risk = risk
                best_attain = attain
                best_effective_attain = eff_attain

        # 进行两阶段优化（仅保留Risk策略）
        two_stage_risk = self.two_stage_search(
            group_df, pi_min=min_attain_rate, metric="R"
        )

        # 调试：检查两阶段结果
        if two_stage_risk is None:
            # print(f"    警告：组内样本数={len(group_df)}, 无法计算两阶段结果")
            # 尝试使用较宽松的参数重新计算
            two_stage_risk = self.two_stage_search(
                group_df, pi_min=max(0.5, min_attain_rate - 0.2), metric="R"
            )
            if two_stage_risk is not None:
                print(
                    f"    信息：组内样本数={len(group_df)}, 使用宽松参数成功计算两阶段结果"
                )
            else:
                print(
                    f"    警告：组内样本数={len(group_df)}, 即使使用宽松参数仍无法计算两阶段结果"
                )

        return {
            "single_stage": {
                "optimal_week": best_time,
                "expected_risk": best_risk,
                "attainment_rate": best_attain,
                "effective_attainment_rate": best_effective_attain,
            },
            "two_stage_R": two_stage_risk,
        }

    def multifactor_grouping_optimization(
        self, df, max_groups=5, min_attain_rate=0.92, measurement_error=0.03
    ):
        """多因素分组优化（使用强制分离的分组策略）"""
        print(
            f"多因素分组优化 (最小达标率: {min_attain_rate:.1%}, 测量误差: {measurement_error:.1%})..."
        )

        # 使用强制分离的分组策略
        best_groups = None
        best_total_risk = np.inf
        best_strategy = ""

        print("尝试强制BMI区间分离的分组策略...")
        groups = self._group_by_forced_separation(
            df, max_groups, min_attain_rate, measurement_error
        )
        if groups is not None:
            total_risk = (
                groups["expected_risk"] * groups["n_patients"]
            ).sum() / groups["n_patients"].sum()
            best_total_risk = total_risk
            best_groups = groups
            best_strategy = "强制区间分离分组"

        self.optimal_groups = best_groups
        print(f"最优分组策略: {best_strategy}, 总体期望风险: {best_total_risk:.4f}")

        return self.optimal_groups

    def _group_by_forced_separation(
        self, df, n_groups, min_attain_rate, measurement_error
    ):
        """使用强制分离的分组方法，确保BMI区间完全不重叠"""
        # 基于BMI排序
        df_sorted = df.sort_values("K").reset_index(drop=True)

        # 计算理想的分组大小
        total_samples = len(df_sorted)
        target_size = total_samples // n_groups

        groups_info = []

        for i in range(n_groups):
            start_idx = i * target_size
            if i == n_groups - 1:  # 最后一组包含剩余所有样本
                end_idx = total_samples
            else:
                end_idx = (i + 1) * target_size

            group_df = df_sorted.iloc[start_idx:end_idx].copy()

            if len(group_df) < 10:
                continue

            # 计算明确的BMI区间边界
            bmi_min = group_df["K"].min()
            bmi_max = group_df["K"].max()

            # 为了确保完全分离，调整边界
            if i < n_groups - 1:
                # 不是最后一组，需要确保与下一组分离
                next_start_idx = (i + 1) * target_size
                if next_start_idx < total_samples:
                    next_min_bmi = df_sorted.iloc[next_start_idx]["K"]
                    # 设置当前组的最大值为当前组最后一个样本的BMI值
                    bmi_max = group_df["K"].iloc[-1]
                    # 确保与下一组有明确分界
                    if bmi_max >= next_min_bmi:
                        # 如果仍有重叠，使用中点作为分界
                        bmi_max = (bmi_max + next_min_bmi) / 2 - 0.01

            # 添加cluster标签用于后续模型拟合
            group_df = group_df.copy()
            group_df["cluster"] = i

            # 优化该组的检测时间
            opt_results = self.optimize_group_testing_time(
                group_df,
                min_attain_rate=min_attain_rate,
                measurement_error=measurement_error,
            )

            group_stats = self.calculate_group_statistics(group_df)

            # 提取单阶段结果用于兼容性
            single_stage = opt_results["single_stage"]
            two_stage_r = opt_results["two_stage_R"]

            groups_info.append(
                {
                    "group_id": i + 1,
                    "bmi_min": bmi_min,
                    "bmi_max": bmi_max,
                    "bmi_range": f"[{bmi_min:.1f}, {bmi_max:.1f}{')'if i < n_groups - 1 else ']'}",
                    # 单阶段结果（保持兼容性）
                    "optimal_week": single_stage["optimal_week"],
                    "expected_risk": single_stage["expected_risk"],
                    "attainment_rate": single_stage["attainment_rate"],
                    "effective_attainment_rate": single_stage[
                        "effective_attainment_rate"
                    ],
                    # 两阶段结果 - Risk策略
                    "two_stage_R_t0": two_stage_r["t0"] if two_stage_r else None,
                    "two_stage_R_t1": two_stage_r["t1"] if two_stage_r else None,
                    "two_stage_R_F0": two_stage_r["F0"] if two_stage_r else None,
                    "two_stage_R_F1": two_stage_r["F1"] if two_stage_r else None,
                    "two_stage_R_retest_rate": (
                        two_stage_r["retest_rate"] if two_stage_r else None
                    ),
                    "two_stage_R_value": two_stage_r["val"] if two_stage_r else None,
                    **group_stats,
                }
            )

        # 为整个数据集创建cluster列用于事件时间模型
        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = -1

        for i, group_info in enumerate(groups_info):
            bmi_min = group_info["bmi_min"]
            bmi_max = group_info["bmi_max"]
            if i < len(groups_info) - 1:  # 不是最后一组
                mask = (df_with_clusters["K"] >= bmi_min) & (
                    df_with_clusters["K"] < bmi_max
                )
            else:  # 最后一组
                mask = (df_with_clusters["K"] >= bmi_min) & (
                    df_with_clusters["K"] <= bmi_max
                )
            df_with_clusters.loc[mask, "cluster"] = i

        # 为每个cluster拟合组级事件时间模型
        self._fit_group_event_models(
            df_with_clusters, id_col="B", time_col="J_week", value_col="V", thr=0.04
        )

        return pd.DataFrame(groups_info) if groups_info else None

    def calculate_group_statistics(self, group_df):
        """计算群体统计特征 - 修正：使用唯一孕妇ID计算人数"""
        # 计算唯一孕妇数量，而不是记录数量
        unique_patients = group_df["B"].nunique()  # 使用孕妇代码计算唯一人数

        stats = {
            "avg_age": group_df["C"].mean(),
            "avg_height": group_df["D"].mean(),
            "avg_weight": group_df["E"].mean(),
            "avg_bmi": group_df["K"].mean(),
            "n_patients": unique_patients,  # 修正：使用唯一孕妇数量
        }
        return stats

    def build_intervals_from_longitudinal(
        self,
        df: pd.DataFrame,
        id_col: str = "A",
        time_col: str = "J_week",
        value_col: str = "V",
        thr: float = 0.04,
    ) -> pd.DataFrame:
        """
        将同一孕妇的多次测量转换为首次达标的区间删失 (L, R]：
        - 若从未达标：右删失 (C, +inf)，此处用 L=C, R=inf, censor='right'
        - 若第一次测到达标为首测：左删失 (-inf, R]
        - 否则：区间删失 (L, R]
        返回列：id, L, R, censor_type
        """
        dfc = df[[id_col, time_col, value_col]].dropna(subset=[id_col, time_col]).copy()
        dfc = dfc.sort_values([id_col, time_col])
        rows = []
        for pid, g in dfc.groupby(id_col):
            weeks = g[time_col].to_numpy(dtype=float)
            vals = g[value_col].to_numpy(dtype=float)
            hit_idx = np.where(vals >= thr)[0]
            if hit_idx.size == 0:
                # 右删失：用最后一次检测周作为C (L=C, R=+inf)
                if weeks.size == 0:
                    continue
                L = weeks[-1]
                R = np.inf
                censor = "right"
            else:
                r = int(hit_idx[0])
                R = weeks[r]
                if r == 0:
                    L = -np.inf
                    censor = "left"
                else:
                    L = weeks[r - 1]
                    censor = "interval"
            rows.append(
                {"id": pid, "L": float(L), "R": float(R), "censor_type": censor}
            )
        return pd.DataFrame(rows)

    # ---------------- 新增：Turnbull NPMLE（简化、数值稳定） ----------------
    def turnbull_npmle(
        self, iv_df: pd.DataFrame, max_iter: int = 3000, tol: float = 1e-10
    ):
        """
        输入 iv_df 包含 L, R, censor_type，返回 dict {support, mass, S, F}
        简化实现：支持点取所有有限端点并用 EM 更新质量
        """
        g = iv_df.copy()
        L = g["L"].to_numpy(dtype=float)
        R = g["R"].to_numpy(dtype=float)
        finite_L = L[np.isfinite(L)]
        finite_R = R[np.isfinite(R)]
        if finite_L.size + finite_R.size == 0:
            return {
                "support": np.array([]),
                "mass": np.array([]),
                "S": np.array([]),
                "F": np.array([]),
            }
        support = np.unique(np.concatenate([finite_L, finite_R]))
        K = len(support)
        n = len(g)
        # 构建 J_i 集合
        J = []
        for Li, Ri in zip(L, R):
            if np.isneginf(Li) and np.isfinite(Ri):
                idx = np.where(support <= Ri)[0]
            elif np.isposinf(Ri) and np.isfinite(Li):
                idx = np.where(support > Li)[0]
            else:
                idx = np.where((support > Li) & (support <= Ri))[0]
            if idx.size == 0:
                near = np.searchsorted(support, np.nan_to_num(Ri, nan=Li))
                near = min(max(0, near - 1), K - 1)
                idx = np.array([near], dtype=int)
            J.append(idx)
        # 初始化均匀分布
        p = np.full(K, 1.0 / K, dtype=float)
        for it in range(max_iter):
            counts = np.zeros(K, dtype=float)
            for idx in J:
                den = p[idx].sum()
                if den <= 0:
                    counts[idx] += 1.0 / len(idx)
                else:
                    counts[idx] += p[idx] / den
            p_new = counts / n
            # 数值修正
            p_new[p_new < 1e-12] = 0.0
            s = p_new.sum()
            if s <= 0:
                p_new = np.full(K, 1.0 / K, dtype=float)
            else:
                p_new = p_new / s
            if np.linalg.norm(p_new - p, ord=1) < tol:
                p = p_new
                break
            p = p_new
        F = np.cumsum(p)
        S = 1.0 - F
        return {"support": support, "mass": p, "S": S, "F": F}

    # ---------------- 新增：阶梯生存函数在任意 t 上求值 ----------------
    def eval_step_S(
        self, t: np.ndarray, support: np.ndarray, S_support: np.ndarray
    ) -> np.ndarray:
        """
        阶梯函数评估：对于 t < support[0] -> S=1；对于 support[j-1] < t <= support[j] -> S=S_support[j]
        """
        if support.size == 0:
            return np.ones_like(t, dtype=float)
        idx = np.searchsorted(support, t, side="right") - 1
        idx = np.clip(idx, -1, len(support) - 1)
        out = np.empty_like(t, dtype=float)
        mask_before = idx < 0
        out[mask_before] = 1.0
        mask_other = ~mask_before
        out[mask_other] = S_support[idx[mask_other]]
        return out

    # ---------------- 新增：对每个聚类组拟合事件时间模型 ----------------
    def _fit_group_event_models(
        self, df_with_cluster, id_col="A", time_col="J_week", value_col="V", thr=0.04
    ):
        """
        对含有 cluster 列的数据，构建组内区间删失并为每组计算 NPMLE，结果保存在 self.group_surv（dict）
        """
        if "cluster" not in df_with_cluster.columns:
            self.group_surv = {}
            return
        iv_all = self.build_intervals_from_longitudinal(
            df_with_cluster,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            thr=thr,
        )
        group_surv = {}
        for g in sorted(df_with_cluster["cluster"].unique()):
            ids = df_with_cluster[df_with_cluster["cluster"] == g][id_col].unique()
            iv_g = iv_all[iv_all["id"].isin(ids)].reset_index(drop=True)
            if iv_g.shape[0] == 0:
                group_surv[g] = {
                    "support": np.array([]),
                    "mass": np.array([]),
                    "S": np.array([]),
                }
                continue
            res = self.turnbull_npmle(iv_g)
            group_surv[g] = res
        self.group_surv = group_surv

    def _group_by_clustering(self, df, n_groups, min_attain_rate, measurement_error):
        """基于多因素的聚类分组 - 改进版本减少重叠"""
        from sklearn.cluster import KMeans
        from sklearn.tree import DecisionTreeRegressor

        # 使用改进的聚类方法，重点解决重叠问题
        # 准备聚类特征，给BMI更高的权重
        cluster_features = df[["K", "C", "D", "E"]].copy()

        # 标准化，但保持BMI的相对重要性
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_features)

        # 给BMI列（第0列）增加权重
        X_scaled[:, 0] *= 3.0  # BMI权重增加3倍

        # 尝试多次聚类，选择BMI分离度最好的结果
        best_labels = None
        best_separation_score = -np.inf

        for random_state in range(42, 52):  # 尝试10个不同的随机种子
            kmeans = KMeans(n_clusters=n_groups, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # 计算BMI分离度分数
            separation_score = self._calculate_bmi_separation_score(
                df, labels, n_groups
            )

            if separation_score > best_separation_score:
                best_separation_score = separation_score
                best_labels = labels.copy()

        df_temp = df.copy()

        # 进一步优化：使用决策树来优化边界
        best_labels = self._optimize_boundaries_with_tree(
            df_temp, best_labels, n_groups
        )

        df_temp["cluster"] = best_labels

        # 为每个cluster拟合组级事件时间模型
        self._fit_group_event_models(
            df_temp, id_col="B", time_col="J_week", value_col="V", thr=0.04
        )

        groups_info = []

        for i in range(n_groups):
            group_df = df_temp[df_temp["cluster"] == i]

            if len(group_df) < 10:
                continue

            # 优化该组的检测时间
            opt_results = self.optimize_group_testing_time(
                group_df,
                min_attain_rate=min_attain_rate,
                measurement_error=measurement_error,
            )

            group_stats = self.calculate_group_statistics(group_df)

            # 提取单阶段结果用于兼容性
            single_stage = opt_results["single_stage"]

            groups_info.append(
                {
                    "group_id": i + 1,
                    "bmi_min": group_df["K"].min(),
                    "bmi_max": group_df["K"].max(),
                    "bmi_range": f"[{group_df['K'].min():.1f}, {group_df['K'].max():.1f}]",
                    "optimal_week": single_stage["optimal_week"],
                    "expected_risk": single_stage["expected_risk"],
                    "attainment_rate": single_stage["attainment_rate"],
                    "effective_attainment_rate": single_stage[
                        "effective_attainment_rate"
                    ],
                    **group_stats,
                }
            )

        return pd.DataFrame(groups_info) if groups_info else None

    def _calculate_bmi_separation_score(self, df, labels, n_groups):
        """计算BMI分组的分离度分数，分数越高表示重叠越少"""
        group_ranges = []

        for i in range(n_groups):
            mask = labels == i
            if mask.sum() > 0:
                bmi_min = df[mask]["K"].min()
                bmi_max = df[mask]["K"].max()
                group_ranges.append((bmi_min, bmi_max))

        if len(group_ranges) < 2:
            return 0

        # 按最小BMI排序
        group_ranges.sort(key=lambda x: x[0])

        # 计算分离度：相邻组之间的间隙 vs 组内范围
        separation_score = 0
        total_overlap = 0

        for i in range(len(group_ranges) - 1):
            current_max = group_ranges[i][1]
            next_min = group_ranges[i + 1][0]

            if current_max > next_min:  # 有重叠
                overlap = current_max - next_min
                total_overlap += overlap
            else:  # 有间隙，这是好的
                gap = next_min - current_max
                separation_score += gap

        # 分离度分数 = 间隙总和 - 重叠惩罚
        return separation_score - total_overlap * 2

    def _optimize_boundaries_with_tree(self, df, initial_labels, n_groups):
        """使用决策树优化分组边界以减少重叠"""
        from sklearn.tree import DecisionTreeClassifier

        # 按BMI对分组重新排序
        group_medians = []
        for i in range(n_groups):
            mask = initial_labels == i
            if mask.sum() > 0:
                median_bmi = df[mask]["K"].median()
                group_medians.append((i, median_bmi))

        group_medians.sort(key=lambda x: x[1])

        # 重新映射标签
        label_mapping = {
            old_id: new_id for new_id, (old_id, _) in enumerate(group_medians)
        }
        ordered_labels = np.array([label_mapping[label] for label in initial_labels])

        # 使用决策树来学习更好的分组边界
        features = df[["K", "C", "D", "E"]].values

        try:
            dt = DecisionTreeClassifier(
                max_depth=3, min_samples_split=20, min_samples_leaf=10, random_state=42
            )
            dt.fit(features, ordered_labels)

            # 使用决策树预测新的标签
            new_labels = dt.predict(features)

            # 验证新标签是否减少了重叠
            old_score = self._calculate_bmi_separation_score(
                df, ordered_labels, n_groups
            )
            new_score = self._calculate_bmi_separation_score(df, new_labels, n_groups)

            if new_score > old_score:
                return new_labels
            else:
                return ordered_labels

        except Exception as e:
            print(f"决策树优化失败，使用原始聚类结果: {e}")
            return ordered_labels

    def _adjust_clusters_for_bmi_separation(self, df, cluster_labels, n_groups):
        """调整聚类结果以减少BMI重叠"""
        df_temp = df.copy()
        df_temp["original_cluster"] = cluster_labels

        # 计算每个聚类的BMI中位数
        cluster_medians = []
        for i in range(n_groups):
            mask = cluster_labels == i
            if mask.sum() > 0:
                median_bmi = df_temp[mask]["K"].median()
                cluster_medians.append((i, median_bmi))

        # 按BMI中位数排序聚类
        cluster_medians.sort(key=lambda x: x[1])

        # 重新分配聚类标签
        new_labels = np.zeros_like(cluster_labels)
        for new_id, (old_id, _) in enumerate(cluster_medians):
            mask = cluster_labels == old_id
            new_labels[mask] = new_id

        # 进一步调整边界以减少重叠
        adjusted_labels = new_labels.copy()

        for i in range(len(cluster_medians) - 1):
            # 找到相邻两组的BMI重叠区域
            group_i_mask = new_labels == i
            group_j_mask = new_labels == i + 1

            if group_i_mask.sum() == 0 or group_j_mask.sum() == 0:
                continue

            group_i_max = df_temp[group_i_mask]["K"].max()
            group_j_min = df_temp[group_j_mask]["K"].min()

            # 如果有重叠，找到合适的分割点
            if group_i_max > group_j_min:
                # 使用两组BMI的加权平均作为分割点
                group_i_median = df_temp[group_i_mask]["K"].median()
                group_j_median = df_temp[group_j_mask]["K"].median()
                split_point = (group_i_median + group_j_median) / 2

                # 重新分配重叠区域的点
                overlap_mask = (df_temp["K"] >= group_j_min) & (
                    df_temp["K"] <= group_i_max
                )
                overlap_indices = df_temp[overlap_mask].index

                for idx in overlap_indices:
                    if df_temp.loc[idx, "K"] <= split_point:
                        adjusted_labels[df_temp.index.get_loc(idx)] = i
                    else:
                        adjusted_labels[df_temp.index.get_loc(idx)] = i + 1

        return adjusted_labels

    def sensitivity_analysis(self, df):
        """敏感性分析"""
        print("进行敏感性分析...")

        # 不同参数组合
        sensitivity_params = [
            {"min_attain_rate": 0.88, "measurement_error": 0.02, "name": "宽松约束"},
            {"min_attain_rate": 0.92, "measurement_error": 0.03, "name": "标准约束"},
            {"min_attain_rate": 0.96, "measurement_error": 0.05, "name": "严格约束"},
            {"min_attain_rate": 0.90, "measurement_error": 0.01, "name": "高精度测量"},
            {"min_attain_rate": 0.94, "measurement_error": 0.07, "name": "保守策略"},
        ]

        sensitivity_results = []

        for params in sensitivity_params:
            print(f"测试{params['name']}...")

            # 重新优化分组
            temp_groups = self.multifactor_grouping_optimization(
                df,
                min_attain_rate=params["min_attain_rate"],
                measurement_error=params["measurement_error"],
            )

            if temp_groups is not None and len(temp_groups) > 0:
                sensitivity_results.append(
                    {
                        "scenario": params["name"],
                        "min_attain_rate": params["min_attain_rate"],
                        "measurement_error": params["measurement_error"],
                        "n_groups": len(temp_groups),
                        "avg_optimal_week": temp_groups["optimal_week"].mean(),
                        "avg_attainment_rate": temp_groups["attainment_rate"].mean(),
                        "total_risk": (
                            temp_groups["expected_risk"] * temp_groups["n_patients"]
                        ).sum()
                        / temp_groups["n_patients"].sum(),
                    }
                )

        return pd.DataFrame(sensitivity_results)

    def analyze_factor_importance(self):
        """分析因素重要性"""
        if self.prob_model is None:
            print("请先拟合概率模型")
            return None

        # 使用随机森林的特征重要性
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.prob_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("多因素重要性排序:")
        print(importance_df.to_string(index=False))

        return importance_df

    def plot_results(self, df):
        """绘制分析结果"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle("多因素BMI分组与两阶段策略优化分析结果", fontsize=16)

        # 1. BMI分组可视化
        axes[0, 0].hist(df["K"], bins=30, alpha=0.7, edgecolor="black")
        if self.optimal_groups is not None:
            for _, group in self.optimal_groups.iterrows():
                axes[0, 0].axvline(
                    group["bmi_min"], color="red", linestyle="--", alpha=0.7
                )
        axes[0, 0].set_xlabel("BMI")
        axes[0, 0].set_ylabel("频次")
        axes[0, 0].set_title("BMI分布及最优分组")

        # 2. 两阶段策略时点对比
        if self.optimal_groups is not None:
            group_ids = range(len(self.optimal_groups))

            # 绘制t0和t1
            has_two_stage = self.optimal_groups["two_stage_R_t0"].notna().any()

            if has_two_stage:
                t0_values = self.optimal_groups["two_stage_R_t0"].fillna(0)
                t1_values = self.optimal_groups["two_stage_R_t1"].fillna(0)

                axes[0, 1].scatter(
                    group_ids,
                    t0_values,
                    color="green",
                    s=100,
                    label="首次检测时间(t0)",
                    marker="o",
                    alpha=0.8,
                )
                axes[0, 1].scatter(
                    group_ids,
                    t1_values,
                    color="red",
                    s=100,
                    label="保底检测时间(t1)",
                    marker="^",
                    alpha=0.8,
                )

                # 连线显示时间窗口
                for i, (t0, t1) in enumerate(zip(t0_values, t1_values)):
                    if pd.notna(t0) and pd.notna(t1):
                        axes[0, 1].plot([i, i], [t0, t1], "k-", alpha=0.3, linewidth=2)
                        axes[0, 1].text(
                            i,
                            (t0 + t1) / 2,
                            f"{t1-t0:.1f}周",
                            ha="center",
                            fontsize=8,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                            ),
                        )

                axes[0, 1].set_ylabel("检测孕周")
                axes[0, 1].set_title("两阶段策略时点分布")
                axes[0, 1].legend()
            else:
                axes[0, 1].plot(
                    self.optimal_groups["avg_bmi"],
                    self.optimal_groups["optimal_week"],
                    "ro-",
                    linewidth=2,
                    markersize=8,
                )
                axes[0, 1].set_xlabel("平均BMI")
                axes[0, 1].set_ylabel("最佳检测孕周")
                axes[0, 1].set_title("最佳检测时点vs BMI")

            axes[0, 1].set_xlabel("分组ID")
            axes[0, 1].set_xticks(group_ids)
            axes[0, 1].set_xticklabels([f"组{i+1}" for i in group_ids])
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 两阶段策略效果对比
        if self.optimal_groups is not None:
            x_pos = range(len(self.optimal_groups))

            # 检查是否有两阶段数据
            has_two_stage = self.optimal_groups["two_stage_R_F0"].notna().any()

            if has_two_stage:
                # 显示早期达标率和最终达标率
                early_rates = self.optimal_groups["two_stage_R_F0"].fillna(0)
                final_rates = self.optimal_groups["two_stage_R_F1"].fillna(0)

                bars1 = axes[0, 2].bar(
                    [x - 0.2 for x in x_pos],
                    early_rates,
                    width=0.4,
                    label="早期达标率(t0)",
                    alpha=0.7,
                    color="lightgreen",
                )
                bars2 = axes[0, 2].bar(
                    [x + 0.2 for x in x_pos],
                    final_rates,
                    width=0.4,
                    label="最终达标率(t1)",
                    alpha=0.7,
                    color="lightcoral",
                )
            else:
                # 回退到原有显示
                bars1 = axes[0, 2].bar(
                    [x - 0.2 for x in x_pos],
                    self.optimal_groups["attainment_rate"],
                    width=0.4,
                    label="实际达标率",
                    alpha=0.7,
                )
                bars2 = axes[0, 2].bar(
                    [x + 0.2 for x in x_pos],
                    self.optimal_groups["effective_attainment_rate"],
                    width=0.4,
                    label="有效达标率",
                    alpha=0.7,
                )

            axes[0, 2].axhline(y=0.9, color="red", linestyle="--", label="90%目标线")
            axes[0, 2].set_xlabel("分组")
            axes[0, 2].set_ylabel("达标率")
            axes[0, 2].set_title(
                "两阶段策略达标率对比" if has_two_stage else "各组达标率对比"
            )
            axes[0, 2].set_xticks(x_pos)
            axes[0, 2].set_xticklabels([f"组{i + 1}" for i in x_pos])
            axes[0, 2].legend()

        # 4. 多因素相关性热图
        factors = ["K", "C", "D", "E", "J_week", "V"]
        available_factors = [f for f in factors if f in df.columns]
        if len(available_factors) > 2:
            corr_matrix = df[available_factors].corr()
            sns.heatmap(
                corr_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[1, 0]
            )
            axes[1, 0].set_title("多因素相关性热图")

        # 5. 特征重要性
        if self.prob_model is not None:
            importance = self.prob_model.feature_importances_
            top_indices = np.argsort(importance)[-10:]

            axes[1, 1].barh(range(len(top_indices)), importance[top_indices])
            axes[1, 1].set_yticks(range(len(top_indices)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in top_indices])
            axes[1, 1].set_xlabel("重要性")
            axes[1, 1].set_title("特征重要性 (Top 10)")

        # 6. 年龄vs达标概率分布
        if len(df) > 50:  # 确保有足够数据
            age_bins = pd.cut(df["C"], bins=5)
            age_attain_rates = df.groupby(age_bins)["V"].apply(
                lambda x: (x >= 0.04).mean()
            )
            age_centers = [
                interval.mid
                for interval in age_attain_rates.index
                if pd.notna(interval.mid)
            ]
            valid_rates = [rate for rate in age_attain_rates.values if pd.notna(rate)]

            if len(age_centers) > 0 and len(valid_rates) > 0:
                axes[1, 2].plot(age_centers, valid_rates, "o-")
                axes[1, 2].set_xlabel("年龄")
                axes[1, 2].set_ylabel("达标率")
                axes[1, 2].set_title("年龄vs达标率")

        # 7. 风险分布
        if self.optimal_groups is not None:
            risks = self.optimal_groups["expected_risk"]
            group_names = [f"组{i + 1}" for i in range(len(risks))]

            bars = axes[2, 0].bar(group_names, risks, alpha=0.7, color="orange")
            axes[2, 0].set_ylabel("期望风险")
            axes[2, 0].set_title("各组期望风险")

            # 标注数值
            for bar, risk in zip(bars, risks):
                axes[2, 0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{risk:.2f}",
                    ha="center",
                    va="bottom",
                )

        # 8. 两阶段策略可视化 (类似notebook中的策略图)
        if (
            self.optimal_groups is not None
            and self.optimal_groups["two_stage_R_t0"].notna().any()
        ):
            # 使用生存函数数据创建类似notebook的策略图
            axes[2, 1].set_title("两阶段策略示意图(前3组)")

            # 为前3组绘制假想的F_g(t)曲线和策略线
            colors = ["blue", "green", "orange"]

            for idx in range(min(3, len(self.optimal_groups))):
                group = self.optimal_groups.iloc[idx]

                if pd.notna(group["two_stage_R_t0"]) and pd.notna(
                    group["two_stage_R_t1"]
                ):
                    t0 = group["two_stage_R_t0"]
                    t1 = group["two_stage_R_t1"]
                    F0 = group["two_stage_R_F0"]
                    F1 = group["two_stage_R_F1"]

                    # 创建示意性的累积达标曲线
                    t_range = np.linspace(10, 25, 100)
                    # 使用sigmoidal函数模拟达标曲线
                    F_curve = F1 / (1 + np.exp(-0.5 * (t_range - (t0 + t1) / 2)))

                    # 绘制达标曲线
                    axes[2, 1].plot(
                        t_range,
                        F_curve,
                        color=colors[idx],
                        linewidth=2,
                        label=f"组{idx+1} F_g(t)",
                    )

                    # 标记关键时点
                    axes[2, 1].axvline(t0, color=colors[idx], linestyle="--", alpha=0.7)
                    axes[2, 1].axvline(t1, color="red", linestyle="--", alpha=0.7)
                    axes[2, 1].scatter([t0], [F0], color=colors[idx], s=80, zorder=5)

                    # 添加标注
                    axes[2, 1].text(
                        t0,
                        F0 + 0.05,
                        f"F={F0:.2f}",
                        ha="center",
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="white", alpha=0.8
                        ),
                    )

            # 添加95%基准线
            axes[2, 1].axhline(
                0.95, color="orange", linestyle=":", alpha=0.7, label="95%目标"
            )
            axes[2, 1].set_xlabel("孕周(周)")
            axes[2, 1].set_ylabel("累积达标率 F_g(t)")
            axes[2, 1].legend(fontsize=8)
            axes[2, 1].set_ylim(0, 1.05)
            axes[2, 1].grid(True, alpha=0.3)
        else:
            # 原有的身高体重分布图（作为后备）
            if len(df) > 0:
                达标_mask = df["V"] >= 0.04

                scatter1 = axes[2, 1].scatter(
                    df[达标_mask]["D"],
                    df[达标_mask]["E"],
                    alpha=0.6,
                    label="达标",
                    c="green",
                    s=30,
                )
                scatter2 = axes[2, 1].scatter(
                    df[~达标_mask]["D"],
                    df[~达标_mask]["E"],
                    alpha=0.6,
                    label="未达标",
                    c="red",
                    s=30,
                )
                axes[2, 1].set_xlabel("身高")
                axes[2, 1].set_ylabel("体重")
                axes[2, 1].set_title("身高体重分布(按达标状态着色)")
                axes[2, 1].legend()

        # 9. 两阶段策略结果汇总表
        if self.optimal_groups is not None:
            axes[2, 2].axis("off")

            table_data = []
            has_two_stage = self.optimal_groups["two_stage_R_t0"].notna().any()

            for _, row in self.optimal_groups.iterrows():
                if has_two_stage and pd.notna(row.get("two_stage_R_t0")):
                    # 包含两阶段信息的表格
                    table_data.append(
                        [
                            f"组{row['group_id']}",
                            row["bmi_range"],
                            f"{row['two_stage_R_t0']:.1f}/{row['two_stage_R_t1']:.1f}",
                            f"{row['two_stage_R_F0']:.2f}",
                            f"{row['two_stage_R_F1']:.2f}",
                            f"{row['n_patients']}",
                        ]
                    )
                else:
                    # 原有的单阶段表格
                    table_data.append(
                        [
                            f"组{row['group_id']}",
                            row["bmi_range"],
                            f"{row['optimal_week']:.1f}",
                            f"{row['attainment_rate']:.2f}",
                            f"{row['effective_attainment_rate']:.2f}",
                            f"{row['n_patients']}",
                        ]
                    )

            if has_two_stage:
                col_labels = [
                    "分组",
                    "BMI范围",
                    "t0/t1(周)",
                    "早期达标率",
                    "最终达标率",
                    "样本数",
                ]
                title = "两阶段策略详细结果"
            else:
                col_labels = [
                    "分组",
                    "BMI范围",
                    "最佳时点",
                    "实际达标率",
                    "有效达标率",
                    "样本数",
                ]
                title = "分组详细结果"

            table = axes[2, 2].table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            axes[2, 2].set_title(title)

        plt.tight_layout()
        return fig

    def plot_two_stage_strategy(self, optimal_groups):
        """绘制两阶段策略可视化图（类似notebook中的策略图）"""
        if optimal_groups is None or not optimal_groups["two_stage_R_t0"].notna().any():
            return None

        groups = optimal_groups[optimal_groups["two_stage_R_t0"].notna()]
        G = len(groups)

        if G == 0:
            return None

        from math import ceil

        ncols = 3 if G > 6 else 2
        nrows = ceil(G / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).ravel()

        fig.suptitle("两阶段NIPT检测策略", fontsize=16, fontweight="bold")

        for i, (_, group) in enumerate(groups.iterrows()):
            if i >= len(axes):
                break

            ax = axes[i]

            # 获取该组数据
            t0 = group["two_stage_R_t0"]
            t_star = group["two_stage_R_t1"]
            F_t0 = group["two_stage_R_F0"]
            F_t1 = group["two_stage_R_F1"]

            # 创建时间网格
            t_grid = np.linspace(10, 25, 100)

            # 使用分段函数模拟F_g(t)曲线
            # 在t0之前缓慢增长，t0处达到F_t0，t_star处达到F_t1
            Fg = np.zeros_like(t_grid)
            for j, t in enumerate(t_grid):
                if t <= t0:
                    # t0之前的线性增长
                    Fg[j] = F_t0 * (t - 10) / (t0 - 10) if t0 > 10 else F_t0
                elif t <= t_star:
                    # t0到t_star之间的线性增长
                    Fg[j] = F_t0 + (F_t1 - F_t0) * (t - t0) / (t_star - t0)
                else:
                    # t_star之后保持不变
                    Fg[j] = F_t1

            # 限制在[0,1]范围内
            Fg = np.clip(Fg, 0, 1)

            # 绘制F_g(t)曲线（阶梯函数风格）
            ax.step(t_grid, Fg, where="post", linewidth=3, label="F_g(t)", color="blue")

            # 标记关键时点
            # t0时点
            ax.axvline(
                t0, color="green", linestyle="--", linewidth=2, label=f"t0={t0:.1f}周"
            )
            ax.plot([t0], [F_t0], "go", markersize=8)
            ax.text(
                t0,
                min(1.0, F_t0 + 0.06),
                f"F={F_t0:.2f}",
                ha="center",
                color="green",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

            # t*_g时点
            ax.axvline(
                t_star,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"t*={t_star:.1f}周",
            )
            ax.plot([t_star], [F_t1], "rs", markersize=8)

            # 95%基准线
            ax.axhline(
                0.95,
                color="orange",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label="95%目标",
            )

            # 设置标题和标签
            bmi_range = (
                group["bmi_range"]
                if "bmi_range" in group
                else f"组{int(group['group_id'])}"
            )
            ax.set_title(
                f'组{int(group["group_id"])}: {bmi_range}\n两阶段策略',
                fontsize=12,
                fontweight="bold",
            )

            if i >= (nrows - 1) * ncols:  # 最底行
                ax.set_xlabel("孕周(周)", fontsize=10)
            if i % ncols == 0:  # 最左列
                ax.set_ylabel("累积达标率 F_g(t)", fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(10, 25)

            # 添加图例（只在第一个图中显示）
            if i == 0:
                ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

            # 添加策略效果文本
            advance_weeks = t_star - t0
            ax.text(
                0.02,
                0.98,
                f"提前: {advance_weeks:.1f}周\n早期成功率: {F_t0:.1%}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            )

        # 关闭多余子图
        for k in range(G, len(axes)):
            axes[k].axis("off")

        plt.tight_layout()
        return fig


def run_problem3():
    """运行问题3分析"""
    print("=" * 60)
    print("问题3：多因素BMI分组与达标比例约束分析")
    print("=" * 60)

    # 初始化数据处理器
    processor = NIPTDataProcessor()

    # 加载数据
    if not processor.load_data(
        r"D:\HP\OneDrive\Desktop\学校\竞赛\数模国赛\CUMCM2025Problems\C题\prenatal-testing\Q3\男胎检测数据_预处理后.csv"
    ):
        print("数据加载失败，请检查文件'男胎检测数据_预处理后.csv'是否存在")
        return None

    male_df = processor.preprocess_male_data()
    if male_df is None or len(male_df) == 0:
        print("男胎数据预处理失败")
        return None

    # --- 插入：构造区间删失并诊断事件时间信息量 ---
    # 去除同一ID同一孕周的重复：保留最大V（或最近一次）
    male_df = (
        male_df.sort_values(["B", "J_week", "V"])
        .groupby(["B", "J_week"], as_index=False)
        .last()
    )

    # 统计具有多次随访的ID数量
    id_counts = male_df["B"].value_counts()  # 使用孕妇代码而不是序号
    n_multi = int((id_counts > 1).sum())
    print(f"具有多次测量的ID数量: {n_multi}")

    # 构建区间删失表并展示前几行
    iv_df = Problem3Solver(processor).build_intervals_from_longitudinal(
        male_df, id_col="B", time_col="J_week", value_col="V", thr=0.04  # 使用孕妇代码
    )
    print("区间删失样例（最多10条）：")
    print(iv_df.head(10).to_string(index=False))

    # 若信息足够，拟合全体 NPMLE 并在若干时间点评估 F(t)
    if iv_df.shape[0] >= 30 and n_multi >= 30:
        solver_tmp = Problem3Solver(processor)
        npmle_all = solver_tmp.turnbull_npmle(iv_df)
        print(
            "NPMLE support_len:",
            len(npmle_all["support"]),
            " mass_sum:",
            float(np.sum(npmle_all["mass"])),
        )
        for t in [12, 16, 20, 24]:
            St = solver_tmp.eval_step_S(
                np.array([t]), npmle_all["support"], npmle_all["S"]
            )[0]
            print(f"F({t}) = {1.0 - St:.3f}")
    else:
        print(
            "警告：纵向信息不足，NPMLE 估计可能不可靠（建议至少若干十个有重复测量的ID）"
        )
    # -----------------------------------------------------

    # 创建求解器
    solver = Problem3Solver(processor)

    # 步骤1：拟合多因素概率模型
    print("\n步骤1：拟合多因素达标概率模型")
    prob_model = solver.fit_probability_model(male_df)

    # 步骤2：多因素分组优化
    print("\n步骤2：多因素分组优化")
    optimal_groups = solver.multifactor_grouping_optimization(
        male_df, min_attain_rate=0.92, measurement_error=0.03
    )

    # 步骤3：敏感性分析
    print("\n步骤3：敏感性分析")
    sensitivity_results = solver.sensitivity_analysis(male_df)

    # 步骤4：分析因素重要性
    print("\n步骤4：分析多因素重要性")
    importance_df = solver.analyze_factor_importance()

    # 步骤5：绘制结果
    print("\n步骤5：生成可视化结果")
    fig = solver.plot_results(male_df)
    plt.savefig("results/problem3_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 生成两阶段策略专门图像
    if optimal_groups is not None and optimal_groups["two_stage_R_t0"].notna().any():
        print("生成两阶段策略可视化图...")
        fig_strategy = solver.plot_two_stage_strategy(optimal_groups)
        if fig_strategy is not None:
            plt.savefig(
                "results/problem3_two_stage_strategy.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

    # 步骤6：保存结果
    print("\n步骤6：保存分析结果")

    if optimal_groups is not None and len(optimal_groups) > 0:
        # 保存最优分组结果
        optimal_groups.to_csv(
            "results/problem3_optimal_groups.csv", index=False, encoding="utf-8-sig"
        )

        # 保存敏感性分析结果
        if len(sensitivity_results) > 0:
            sensitivity_results.to_csv(
                "results/problem3_sensitivity.csv", index=False, encoding="utf-8-sig"
            )

        # 保存特征重要性
        if importance_df is not None:
            importance_df.to_csv(
                "results/problem3_feature_importance.csv",
                index=False,
                encoding="utf-8-sig",
            )

        # 保存多因素分析详细数据
        multifactor_analysis = []
        for _, group in optimal_groups.iterrows():
            multifactor_analysis.append(
                {
                    "group_id": group["group_id"],
                    "bmi_range": group["bmi_range"],
                    "optimal_week": group["optimal_week"],
                    "attainment_rate": group["attainment_rate"],
                    "effective_attainment_rate": group["effective_attainment_rate"],
                    "expected_risk": group["expected_risk"],
                    "avg_age": group["avg_age"],
                    "avg_height": group["avg_height"],
                    "avg_weight": group["avg_weight"],
                    "avg_bmi": group["avg_bmi"],
                    "n_patients": group["n_patients"],
                }
            )

        multifactor_df = pd.DataFrame(multifactor_analysis)
        multifactor_df.to_csv(
            "results/problem3_multifactor_analysis.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("=" * 60)
        print("问题3分析完成！")
        print("生成的文件：")
        print("- results/problem3_analysis.png (综合分析结果)")
        print("- results/problem3_two_stage_strategy.png (两阶段策略可视化)")
        print("- results/problem3_optimal_groups.csv (最优分组结果)")
        print("- results/problem3_sensitivity.csv (敏感性分析)")
        print("- results/problem3_feature_importance.csv (特征重要性)")
        print("- results/problem3_multifactor_analysis.csv (多因素详细分析)")
        print("- results/problem3_two_stage_analysis.csv (两阶段策略分析)")
        print("- results/problem3_personalized_recommendations.csv (个性化检测建议)")
        print("=" * 60)

        # 输出关键结果摘要
        # 生成两阶段策略分析报告
        two_stage_summary = []
        for _, group in optimal_groups.iterrows():
            if group.get("two_stage_R_t0") is not None:
                # 计算提前周数
                advance_weeks = group["two_stage_R_t1"] - group["two_stage_R_t0"]

                # 计算期望检测时间
                expected_time = group["two_stage_R_t0"] * group[
                    "two_stage_R_F0"
                ] + group["two_stage_R_t1"] * (1 - group["two_stage_R_F0"])

                # 风险评级
                if group["two_stage_R_F0"] >= 0.8:
                    risk_level = "推荐"
                    advice = "早期检测成功率高"
                elif group["two_stage_R_F0"] >= 0.6:
                    risk_level = "建议"
                    advice = "早期检测中等成功率，需关注复测"
                else:
                    risk_level = "谨慎"
                    advice = "早期检测成功率较低，强化复测方案"

                two_stage_summary.append(
                    {
                        "组号": int(group["group_id"]),
                        "BMI区间": group["bmi_range"],
                        "样本量": int(group["n_patients"]),
                        "最优t0(周)": round(float(group["two_stage_R_t0"]), 1),
                        "保底t1(周)": round(float(group["two_stage_R_t1"]), 1),
                        "早期达标率": f"{float(group['two_stage_R_F0']):.1%}",
                        "期望检测时间(周)": round(float(expected_time), 2),
                        "风险评级": risk_level,
                        "临床建议": advice,
                        "提前周数": round(float(advance_weeks), 1),
                    }
                )

        if two_stage_summary:
            print("\n=== 两阶段联合优化结果 ===")
            two_stage_df = pd.DataFrame(two_stage_summary)
            print(two_stage_df.to_string(index=False))

            # 保存两阶段分析结果
            two_stage_df.to_csv(
                "results/problem3_two_stage_analysis.csv",
                index=False,
                encoding="utf-8-sig",
            )

        print("\n=== 关键结果摘要 ===")
        print(f"总样本数: {len(male_df)}")
        print(f"最优分组数: {len(optimal_groups)}")

        print("\n最优多因素分组方案:")
        for _, group in optimal_groups.iterrows():
            print(
                f"组{group['group_id']}: BMI {group['bmi_range']}, "
                f"最佳检测时点 {group['optimal_week']:.1f}周"
            )
            print(
                f"    实际达标率 {group['attainment_rate']:.2f}, "
                f"有效达标率 {group['effective_attainment_rate']:.2f}"
            )
            print(
                f"    平均年龄 {group['avg_age']:.1f}岁, "
                f"平均身高 {group['avg_height']:.1f}cm, "
                f"平均体重 {group['avg_weight']:.1f}kg"
            )

            # 添加两阶段结果展示（仅Risk策略）
            if group.get("two_stage_R_t0") is not None:
                print(
                    f"    两阶段策略: t0={group['two_stage_R_t0']:.1f}周, t1={group['two_stage_R_t1']:.1f}周"
                )
                print(
                    f"        首测达标率: {group['two_stage_R_F0']:.3f}, 复测达标率: {group['two_stage_R_F1']:.3f}"
                )
                print(
                    f"        复测率: {group['two_stage_R_retest_rate']:.3f}, Risk值: {group['two_stage_R_value']:.3f}"
                )
            print()

        print(f"平均最佳检测时点: {optimal_groups['optimal_week'].mean():.1f}周")
        print(f"整体达标率: {optimal_groups['attainment_rate'].mean():.3f}")
        print(
            f"总体期望风险: {(optimal_groups['expected_risk'] * optimal_groups['n_patients']).sum() / optimal_groups['n_patients'].sum():.4f}"
        )

        if len(sensitivity_results) > 0:
            print("\n敏感性分析结果:")
            print(sensitivity_results.to_string(index=False))

        if importance_df is not None and len(importance_df) > 0:
            print(f"\n最重要的影响因素:")
            for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
                print(f"{i + 1}. {row['feature']}: {row['importance']:.4f}")

        # 生成个性化检测方案建议（基于两阶段结果）
        if two_stage_summary:
            print("\n=== 个性化检测方案建议 ===")
            recommendations = []
            for item in two_stage_summary:
                # 时间窗口建议
                t0 = item["最优t0(周)"]
                if t0 < 12:
                    timing_advice = f"在{t0:.1f}周进行首次检测(早期窗口)"
                elif t0 < 16:
                    timing_advice = f"在{t0:.1f}周进行首次检测(标准窗口)"
                else:
                    timing_advice = f"在{t0:.1f}周进行首次检测(延后窗口)"

                full_advice = f"{timing_advice}，{item['临床建议']}"

                recommendations.append(
                    {
                        "组别": f"组{item['组号']}",
                        "BMI范围": item["BMI区间"],
                        "样本特征": f"N={item['样本量']}",
                        "推荐等级": item["风险评级"],
                        "首次检测时间": f"{item['最优t0(周)']}周",
                        "保底检测时间": f"{item['保底t1(周)']}周",
                        "早期成功率": item["早期达标率"],
                        "临床建议": full_advice,
                        "备注": f"预期节省{item['提前周数']}周检测时间",
                    }
                )

            recommendations_df = pd.DataFrame(recommendations)
            print(recommendations_df.to_string(index=False))

            # 保存个性化建议
            recommendations_df.to_csv(
                "results/problem3_personalized_recommendations.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # 生成临床实施建议
            print("\n=== 临床实施建议 ===")

            avg_early_rate = sum(
                [
                    float(item["早期达标率"].rstrip("%")) / 100
                    for item in two_stage_summary
                ]
            ) / len(two_stage_summary)
            avg_advance_weeks = sum(
                [item["提前周数"] for item in two_stage_summary]
            ) / len(two_stage_summary)

            clinical_advice = f"""
1. 个性化检测策略：
   - 根据孕妇BMI分组，采用不同的两阶段检测方案
   - 平均早期检测成功率: {avg_early_rate:.1%}
   - 平均节省检测时间: {avg_advance_weeks:.1f}周

2. 质量控制要点：
   - 确保早期检测的技术可靠性
   - 建立完善的复测跟踪机制  
   - 设置风险阈值预警系统

3. 成本效益考量：
   - 早期检测可显著减少整体检测时间
   - 需平衡检测成本与临床效果
   - 建议建立动态成本调节机制

4. 实施监控：
   - 定期评估各组实际达标率
   - 监控成本控制效果
   - 根据实际情况调整参数

5. 分组策略特点：
   - 低BMI组: 早期检测时点可适当提前，成功率通常较高
   - 高BMI组: 需要更谨慎的检测时点选择，加强保底检测
   - 建议定期根据实际数据调整分组阈值
"""

            print(clinical_advice)

    else:
        print("警告：未能生成有效的分组结果")

    return solver


if __name__ == "__main__":
    run_problem3()
