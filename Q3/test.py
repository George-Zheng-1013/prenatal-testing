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
        """加载Excel数据"""
        try:
            # 读取男胎和女胎数据
            self.male_data = pd.read_excel(file_path, sheet_name="男胎检测数据")
            self.female_data = pd.read_excel(file_path, sheet_name="女胎检测数据")

            # 重命名列为标准格式
            columns = [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                "AA",
                "AB",
                "AC",
                "AD",
                "AE",
            ]

            self.male_data.columns = columns[: len(self.male_data.columns)]
            self.female_data.columns = columns[: len(self.female_data.columns)]

            print(
                f"成功加载数据: 男胎{len(self.male_data)}条, 女胎{len(self.female_data)}条"
            )
            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def parse_gestational_week(self, week_str):
        """解析孕周格式 (如 '11w+6' -> 11.857)"""
        if pd.isna(week_str) or week_str == "":
            return np.nan

        try:
            week_str = str(week_str).strip()
            if "w" in week_str:
                parts = week_str.split("w")
                weeks = float(parts[0])
                if "+" in parts[1]:
                    days = float(parts[1].replace("+", ""))
                    return weeks + days / 7.0
                else:
                    return weeks
            else:
                return float(week_str)
        except:
            return np.nan

    def preprocess_male_data(self):
        """预处理男胎数据"""
        if self.male_data is None:
            print("请先加载数据")
            return None

        df = self.male_data.copy()

        # 转换孕周格式
        df["J_week"] = df["J"].apply(self.parse_gestational_week)

        # 质量控制过滤
        original_len = len(df)

        # 1. 过滤极端GC含量
        df = df[(df["P"] >= 0.35) & (df["P"] <= 0.65)]

        # 2. 过滤低质量测序数据
        df = df[df["L"] >= 1000000]  # 至少100万读段
        df = df[df["AA"] <= 0.5]  # 过滤比例不超过50%

        # 3. 过滤缺失的核心变量
        df = df.dropna(subset=["V", "K", "J_week"])

        # 4. Y染色体浓度范围过滤
        df = df[(df["V"] >= 0) & (df["V"] <= 1)]

        # 5. BMI合理范围
        df = df[(df["K"] >= 15) & (df["K"] <= 50)]

        # 6. 孕周范围
        df = df[(df["J_week"] >= 8) & (df["J_week"] <= 30)]

        print(f"男胎数据预处理: {original_len} -> {len(df)}条记录")
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
            n_estimators=150,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=5,
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
        """计算群体在指定孕周的达标比例"""
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
                "early": 1.0,
                "mid": 3.0,
                "late": 5.0,
                "retest": 2.0,
                "low_attain": 50.0,
                "error": 5.0,
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

    def optimize_group_testing_time(
        self, group_df, time_range=(10, 25), min_attain_rate=0.9, measurement_error=0.05
    ):
        """优化单个组的最佳检测时间"""
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

        return best_time, best_risk, best_attain, best_effective_attain

    def multifactor_grouping_optimization(
        self, df, max_groups=4, min_attain_rate=0.9, measurement_error=0.05
    ):
        """多因素分组优化（只使用聚类分组）"""
        print(
            f"多因素分组优化 (最小达标率: {min_attain_rate:.1%}, 测量误差: {measurement_error:.1%})..."
        )

        # 仅使用多因素聚类分组策略
        best_groups = None
        best_total_risk = np.inf
        best_strategy = ""

        print("尝试多因素聚类分组策略...")
        groups = self._group_by_clustering(
            df, max_groups, min_attain_rate, measurement_error
        )
        if groups is not None:
            total_risk = (
                groups["expected_risk"] * groups["n_patients"]
            ).sum() / groups["n_patients"].sum()
            best_total_risk = total_risk
            best_groups = groups
            best_strategy = "多因素聚类分组"

        self.optimal_groups = best_groups
        print(f"最优分组策略: {best_strategy}, 总体期望风险: {best_total_risk:.4f}")

        return self.optimal_groups

    def calculate_group_statistics(self, group_df):
        """计算群体统计特征"""
        stats = {
            "avg_age": group_df["C"].mean(),
            "avg_height": group_df["D"].mean(),
            "avg_weight": group_df["E"].mean(),
            "avg_bmi": group_df["K"].mean(),
            "n_patients": len(group_df),
        }
        return stats

    def _group_by_clustering(self, df, n_groups, min_attain_rate, measurement_error):
        """基于多因素的聚类分组"""
        from sklearn.cluster import KMeans

        # 准备聚类特征
        cluster_features = df[["K", "C", "D", "E"]].copy()  # BMI, 年龄, 身高, 体重

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_features)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        df_temp = df.copy()
        df_temp["cluster"] = kmeans.fit_predict(X_scaled)

        groups_info = []

        for i in range(n_groups):
            group_df = df_temp[df_temp["cluster"] == i]

            if len(group_df) < 10:
                continue

            # 优化该组的检测时间
            opt_time, opt_risk, opt_attain, opt_eff_attain = (
                self.optimize_group_testing_time(
                    group_df,
                    min_attain_rate=min_attain_rate,
                    measurement_error=measurement_error,
                )
            )

            group_stats = self.calculate_group_statistics(group_df)

            groups_info.append(
                {
                    "group_id": i + 1,
                    "bmi_min": group_df["K"].min(),
                    "bmi_max": group_df["K"].max(),
                    "bmi_range": f"[{group_df['K'].min():.1f}, {group_df['K'].max():.1f}]",
                    "optimal_week": opt_time,
                    "expected_risk": opt_risk,
                    "attainment_rate": opt_attain,
                    "effective_attainment_rate": opt_eff_attain,
                    **group_stats,
                }
            )

        return pd.DataFrame(groups_info) if groups_info else None

    def sensitivity_analysis(self, df):
        """敏感性分析"""
        print("进行敏感性分析...")

        # 不同参数组合
        sensitivity_params = [
            {"min_attain_rate": 0.85, "measurement_error": 0.03, "name": "宽松约束"},
            {"min_attain_rate": 0.90, "measurement_error": 0.05, "name": "标准约束"},
            {"min_attain_rate": 0.95, "measurement_error": 0.08, "name": "严格约束"},
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
        fig.suptitle("多因素BMI分组与达标比例约束分析结果", fontsize=16)

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

        # 2. 最优时点vs BMI
        if self.optimal_groups is not None:
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
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 达标率vs BMI
        if self.optimal_groups is not None:
            x_pos = range(len(self.optimal_groups))
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
            axes[0, 2].set_title("各组达标率对比")
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

        # 8. 身高体重分布 (按达标状态着色)
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

        # 9. 分组结果汇总表
        if self.optimal_groups is not None:
            axes[2, 2].axis("off")

            table_data = []
            for _, row in self.optimal_groups.iterrows():
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

            table = axes[2, 2].table(
                cellText=table_data,
                colLabels=[
                    "分组",
                    "BMI范围",
                    "最佳时点",
                    "实际达标率",
                    "有效达标率",
                    "样本数",
                ],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            axes[2, 2].set_title("分组详细结果")

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
        r"D:\HP\OneDrive\Desktop\学校\竞赛\数模国赛\CUMCM2025Problems\C题\附件.xlsx"
    ):
        print("数据加载失败，请检查文件'附件.xlsx'是否存在")
        return None

    male_df = processor.preprocess_male_data()
    if male_df is None or len(male_df) == 0:
        print("男胎数据预处理失败")
        return None

    # 创建求解器
    solver = Problem3Solver(processor)

    # 步骤1：拟合多因素概率模型
    print("\n步骤1：拟合多因素达标概率模型")
    prob_model = solver.fit_probability_model(male_df)

    # 步骤2：多因素分组优化
    print("\n步骤2：多因素分组优化")
    optimal_groups = solver.multifactor_grouping_optimization(
        male_df, min_attain_rate=0.9, measurement_error=0.05
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
        print("- results/problem3_analysis.png (可视化结果)")
        print("- results/problem3_optimal_groups.csv (最优分组结果)")
        print("- results/problem3_sensitivity.csv (敏感性分析)")
        print("- results/problem3_feature_importance.csv (特征重要性)")
        print("- results/problem3_multifactor_analysis.csv (多因素详细分析)")
        print("=" * 60)

        # 输出关键结果摘要
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

    else:
        print("警告：未能生成有效的分组结果")

    return solver


if __name__ == "__main__":
    run_problem3()
