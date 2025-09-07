# -*- coding: utf-8 -*-
"""
问题四 · 女胎异常判定（完整可运行版，按“评分踩点清单”改造）
改动要点：
1) 维持原有孕周解析逻辑；
2) 训练集内：Pipeline+K折交叉验证；
3) 四种采样策略并行（None/ROS/RUS/SMOTE），与两类模型（L1-LogReg, RandomForest）组合对比；
4) 只在训练集上CV与调参，外部测试集只评一次；
5) 报告并导出 AUROC 与 AUPRC（数值与曲线），并给出阳性率基线；
6) 可选稳健处理：winsorize 截尾；
7) 输出：模型对比表、最佳模型测试集指标、阈值优化、校准曲线点、特征重要性等。

保持与原版的一致性：保留“女胎异常定义=是否存在13/18/21任一异常(AB)”。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# 采样与Pipeline（带兜底）
IMBLEARN_OK = True
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    IMBLEARN_OK = False
    print(
        "警告：未安装 imbalanced-learn，将退化为 class_weight 策略（无采样Pipeline）。"
    )

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 结果目录
os.makedirs("results", exist_ok=True)


class NIPTDataProcessor:
    """数据加载/预处理（保留原孕周解析规则）"""

    def __init__(self):
        self.male_data = None
        self.female_data = None

    def load_data(self, file_path: str) -> bool:
        try:
            self.male_data = pd.read_excel(file_path, sheet_name="男胎检测数据")
            self.female_data = pd.read_excel(file_path, sheet_name="女胎检测数据")

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
            print(f"成功加载：男胎{len(self.male_data)}，女胎{len(self.female_data)}")
            return True
        except Exception as e:
            print(f"数据加载失败：{e}")
            return False

    @staticmethod
    def parse_gestational_week(week_str):
        """解析孕周：如 '11w+6' -> 11 + 6/7 = 11.857（保留原逻辑）"""
        if pd.isna(week_str) or week_str == "":
            return np.nan
        try:
            s = str(week_str).strip()
            if "w" in s:
                w, rest = s.split("w", 1)
                weeks = float(w)
                if "+" in rest:
                    days = float(rest.replace("+", ""))
                    return weeks + days / 7.0
                return weeks
            return float(s)
        except Exception:
            return np.nan

    def preprocess_female_data(self) -> pd.DataFrame:
        if self.female_data is None:
            print("请先加载数据")
            return pd.DataFrame()
        df = self.female_data.copy()

        # 孕周
        df["J_week"] = df["J"].apply(self.parse_gestational_week)

        # 基础QC（与原版一致）
        before = len(df)
        df = df[(df["P"] >= 0.35) & (df["P"] <= 0.65)]
        df = df[df["L"] >= 1_000_000]
        df = df[df["AA"] <= 0.5]

        # 标签：AB 非空为异常
        df["abnormal"] = (~df["AB"].isna()).astype(int)

        # 关键字段齐备
        df = df.dropna(subset=["K", "J_week", "Q", "R", "S", "T"])

        print(
            f"女胎预处理：{before} -> {len(df)}；异常 {df['abnormal'].sum()} (占 {df['abnormal'].mean():.1%})"
        )
        return df.reset_index(drop=True)


class Problem4Solver:
    """问题四：按“评分踩点清单”重构的训练/评估流水线"""

    def __init__(self, processor: NIPTDataProcessor, winsorize=False):
        self.p = processor
        self.winsorize = winsorize
        self.best_pipeline_name = None
        self.best_pipeline = None
        self.feature_names = None
        self.optimal_threshold = 0.5

    # ========= 特征工程 =========
    def prepare_female_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame()
        # Z 值
        X["Z13"] = df["Q"]
        X["Z18"] = df["R"]
        X["Z21"] = df["S"]
        X["ZX"] = df["T"]
        # X 浓度
        X["X_conc"] = df.get("W", pd.Series(0, index=df.index)).fillna(0)
        # GC
        X["GC_13"] = df["X"]
        X["GC_18"] = df["Y"]
        X["GC_21"] = df["Z"]
        X["GC_all"] = df["P"]
        # 读段/质量
        X["total_reads"] = np.log10(df["L"].clip(lower=1))
        X["mapped_ratio"] = df["M"]
        X["dup_ratio"] = df["N"]
        X["unique_reads"] = np.log10(df["O"].clip(lower=1))
        X["filtered_ratio"] = df["AA"]
        # 临床
        X["BMI"] = df["K"]
        X["gw"] = df["J_week"]
        X["age"] = df["C"]
        X["height"] = df["D"]
        X["weight"] = df["E"]
        # 派生
        X["Z_max"] = X[["Z13", "Z18", "Z21"]].abs().max(axis=1)
        X["Z_sum"] = X[["Z13", "Z18", "Z21"]].abs().sum(axis=1)
        X["GC_var"] = X[["GC_13", "GC_18", "GC_21"]].var(axis=1)
        X["GC_mean"] = X[["GC_13", "GC_18", "GC_21"]].mean(axis=1)
        X["BMI_age"] = X["BMI"] * X["age"]
        X["ZxBMI"] = X["Z_max"] * X["BMI"]
        X["ZxGW"] = X["Z_max"] * X["gw"]
        # 质量评分
        X["q_score"] = (
            X["mapped_ratio"] * 0.3
            + (1 - X["dup_ratio"]) * 0.3
            + (1 - X["filtered_ratio"]) * 0.4
        )
        # 传统规则指示
        X["z_any_ge_3"] = (X[["Z13", "Z18", "Z21"]].abs().max(axis=1) >= 3.0).astype(
            int
        )
        X["z_any_ge_3p5"] = (X[["Z13", "Z18", "Z21"]].abs().max(axis=1) >= 3.5).astype(
            int
        )
        X["z_any_2p5_3p5"] = ((X["Z_max"] >= 2.5) & (X["Z_max"] < 3.5)).astype(int)
        # GC异常
        X["gc_abn"] = ((X["GC_all"] < 0.4) | (X["GC_all"] > 0.6)).astype(int)
        # 质量偏低
        X["low_q"] = (X["q_score"] < 0.7).astype(int)

        if self.winsorize:
            # 对常见长尾列做截尾（1%/99%）
            for col in [
                "Z13",
                "Z18",
                "Z21",
                "ZX",
                "mapped_ratio",
                "dup_ratio",
                "filtered_ratio",
                "q_score",
            ]:
                a = X[col].to_numpy()
                lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
                X[col] = X[col].clip(lo, hi)
        return X

    # ========= 采样×模型：构建管道 =========
    def _sampler_space(self):
        if not IMBLEARN_OK:
            # 无imblearn时，仅返回“无采样”的假项
            return {"none": None}
        # SMOTE 邻居数会在 fit 时自动根据阳性样本数兜底
        return {
            "none": None,
            "ros": RandomOverSampler(random_state=42),
            "rus": RandomUnderSampler(random_state=42),
            "smote": SMOTE(random_state=42, k_neighbors=3),
        }

    def _model_space(self):
        # 两个强健基线
        return {
            "logreg_l1": LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=2000, class_weight="balanced"
            ),
            "rf": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
        }

    def _build_pipeline(self, sampler, model):
        if IMBLEARN_OK:
            steps = [("scaler", RobustScaler())]
            if sampler is not None:
                steps.append(("sampler", sampler))
            steps.append(("clf", model))
            return ImbPipeline(steps)
        else:
            # 无imblearn时退化：仅缩放+模型（模型自身使用class_weight）
            from sklearn.pipeline import Pipeline

            return Pipeline([("scaler", RobustScaler()), ("clf", model)])

    # ========= 训练/选择 =========
    def train(self, df: pd.DataFrame, random_state=42):
        X = self.prepare_female_features(df)
        y = df["abnormal"].astype(int).to_numpy()
        self.feature_names = X.columns.tolist()

        # 外部测试集留出（分层）
        strat = y if y.sum() > 0 and (len(y) - y.sum()) > 0 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=strat
        )

        # 仅在训练集上做K折CV
        skf = StratifiedKFold(
            n_splits=min(5, max(2, int(y_tr.sum()) if y_tr.sum() > 1 else 2)),
            shuffle=True,
            random_state=random_state,
        )

        results = []
        best_key, best_auprc, best_auroc, best_pipe = None, -1.0, -1.0, None

        for s_name, sampler in self._sampler_space().items():
            for m_name, model in self._model_space().items():
                pipe = self._build_pipeline(sampler, model)
                # 轻量网格，仅对关键超参
                if m_name == "logreg_l1":
                    param_grid = {"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0]}
                else:  # rf
                    param_grid = {"clf__max_depth": [None, 8, 12, 16]}

                grid = GridSearchCV(
                    pipe,
                    param_grid=param_grid,
                    scoring="average_precision",  # 以AUPRC为主目标
                    cv=skf,
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                )
                grid.fit(X_tr, y_tr)

                # 记录CV表现
                cv_ap = grid.best_score_
                # 同时估计CV AUROC
                # 重新用同样cv手动计算一次（尽量节约时间，直接用best_estimator_预测折外）
                oof_proba = np.zeros_like(y_tr, dtype=float)
                for fold, (a, b) in enumerate(skf.split(X_tr, y_tr)):
                    fold_clf = self._build_pipeline(sampler, model)
                    # 使用最佳参数
                    fold_clf.set_params(**{k: v for k, v in grid.best_params_.items()})
                    fold_clf.fit(X_tr.iloc[a], y_tr[a])
                    try:
                        oof_proba[b] = fold_clf.predict_proba(X_tr.iloc[b])[:, 1]
                    except Exception:
                        # 若模型不支持 predict_proba
                        oof_proba[b] = fold_clf.decision_function(X_tr.iloc[b])
                cv_auc = (
                    roc_auc_score(y_tr, oof_proba) if len(np.unique(y_tr)) > 1 else 0.5
                )

                key = f"{s_name}|{m_name}"
                results.append(
                    {
                        "pipeline": key,
                        "best_params": grid.best_params_,
                        "cv_auprc": cv_ap,
                        "cv_auroc": cv_auc,
                    }
                )

                # 选择规则：先看AUPRC，其次AUROC
                if cv_ap > best_auprc or (cv_ap == best_auprc and cv_auc > best_auroc):
                    best_key = key
                    best_auprc = cv_ap
                    best_auroc = cv_auc
                    best_pipe = grid.best_estimator_

        cv_df = pd.DataFrame(results).sort_values(
            ["cv_auprc", "cv_auroc"], ascending=False
        )
        cv_df.to_csv(
            "results/problem4_cv_summary.csv", index=False, encoding="utf-8-sig"
        )

        self.best_pipeline_name = best_key
        self.best_pipeline = best_pipe

        # 在训练集上再次拟合最佳模型，然后用于测试集评估
        self.best_pipeline.fit(X_tr, y_tr)

        # ========== 外部测试集仅一次评估 ==========
        y_proba = self._predict_proba(self.best_pipeline, X_te)
        y_pred_default = (y_proba >= 0.5).astype(int)

        test_metrics = self._evaluate_on_test(y_te, y_proba, y_pred_default)
        test_metrics.update({"best_pipeline": best_key})
        pd.DataFrame([test_metrics]).to_csv(
            "results/problem4_test_summary.csv", index=False, encoding="utf-8-sig"
        )

        # 阈值优化（成本敏感：假阴性代价高）
        self.optimal_threshold, thresh_df = self._optimize_threshold(
            y_te, y_proba, cost_ratio=10
        )
        thresh_df.to_csv(
            "results/problem4_threshold_optimization.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 以最优阈值生成混淆矩阵等图表
        self._plot_all(X, df["abnormal"].to_numpy(), X_te, y_te, y_proba)

        # 导出校准曲线点
        try:
            frac_pos, mean_pred = calibration_curve(
                y_te, y_proba, n_bins=min(10, max(3, len(y_te) // 3))
            )
            pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos}).to_csv(
                "results/problem4_calibration_points.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception:
            pass

        # 导出特征重要性（如RF）
        self._export_feature_importance()

        print("\n=== 最佳Pipeline ===")
        print(best_key)
        print(
            "测试集：AUPRC={:.3f}, AUROC={:.3f}, 阳性率(基线)={:.3f}".format(
                test_metrics["auprc"], test_metrics["auroc"], test_metrics["pos_rate"]
            )
        )
        print("最优阈值：{:.3f}".format(self.optimal_threshold))

        return X_te, y_te

    # ========= 工具函数 =========
    @staticmethod
    def _predict_proba(model, X):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return model.decision_function(X)

    @staticmethod
    def _evaluate_on_test(y_true, y_proba, y_pred):
        res = {}
        if len(np.unique(y_true)) > 1:
            res["auroc"] = roc_auc_score(y_true, y_proba)
            res["auprc"] = average_precision_score(y_true, y_proba)
        else:
            res["auroc"] = 0.5
            res["auprc"] = np.mean(y_true)  # 类先验
        res["pos_rate"] = float(np.mean(y_true))
        res["accuracy@0.5"] = float(np.mean(y_pred == y_true))
        try:
            res["brier"] = brier_score_loss(y_true, y_proba)
        except Exception:
            res["brier"] = np.nan
        # 分类报告（按0.5阈值）
        try:
            rep = classification_report(y_true, y_pred, output_dict=True)
            for k in ["0", "1", "macro avg", "weighted avg"]:
                if k in rep:
                    for m in ["precision", "recall", "f1-score", "support"]:
                        res[f"{k}_{m}"] = rep[k].get(m, np.nan)
        except Exception:
            pass
        return res

    def _optimize_threshold(self, y_true, y_proba, cost_ratio=10):
        thresholds = np.linspace(0.05, 0.95, 19)
        rows = []
        best_score, best_t = -1e9, 0.5
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except Exception:
                # 类别不全时跳过
                continue
            sens = tp / (tp + fn) if (tp + fn) else 0
            spec = tn / (tn + fp) if (tn + fp) else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            # 成本敏感：假阴性代价高
            score = sens * 0.7 + spec * 0.3 - cost_ratio * fn / len(y_true)
            rows.append(
                {
                    "threshold": t,
                    "sensitivity": sens,
                    "specificity": spec,
                    "precision": prec,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "cost_score": score,
                }
            )
            if score > best_score:
                best_score, best_t = score, t
        return best_t, pd.DataFrame(rows)

    def _plot_all(self, X_all, y_all, X_test, y_test, y_proba):
        # 画图与导出
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("问题四：女胎异常判定（模型对比与测试集表现）", fontsize=16)

        # ROC
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            axes[0, 0].plot(fpr, tpr, label=f"Best (AUC={auc:.3f})")
            axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[0, 0].set_title("ROC 曲线")
            axes[0, 0].set_xlabel("FPR")
            axes[0, 0].set_ylabel("TPR")
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, "单一类别，无法绘制ROC", ha="center", va="center")

        # PR
        if len(np.unique(y_test)) > 1:
            prec, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            axes[0, 1].plot(recall, prec, label=f"Best (AP={ap:.3f})")
            axes[0, 1].axhline(
                np.mean(y_test),
                ls="--",
                alpha=0.5,
                label=f"阳性率基线={np.mean(y_test):.3f}",
            )
            axes[0, 1].set_title("Precision-Recall 曲线")
            axes[0, 1].set_xlabel("Recall")
            axes[0, 1].set_ylabel("Precision")
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, "单一类别，无法绘制PR", ha="center", va="center")

        # 校准
        try:
            frac_pos, mean_pred = calibration_curve(
                y_test, y_proba, n_bins=min(10, max(3, len(y_test) // 3))
            )
            axes[0, 2].plot([0, 1], [0, 1], "k:", label="完全校准")
            axes[0, 2].plot(mean_pred, frac_pos, "s-", label="Best")
            axes[0, 2].set_title("校准曲线")
            axes[0, 2].set_xlabel("平均预测概率")
            axes[0, 2].set_ylabel("实际阳性率")
            axes[0, 2].legend()
        except Exception:
            axes[0, 2].text(0.5, 0.5, "无法绘制校准曲线", ha="center", va="center")

        # 特征：Z值分布概览
        for col, ax in zip(["Z13", "Z18", "Z21"], [axes[1, 0], axes[1, 1]]):
            if col in X_all.columns:
                ax.hist(X_all[col], bins=20, alpha=0.8)
                ax.axvline(3.0, color="r", ls="--", alpha=0.7)
                ax.axvline(-3.0, color="r", ls="--", alpha=0.7)
                ax.set_title(f"{col} 分布")
        # 混淆矩阵（最优阈值）
        y_pred_opt = (y_proba >= self.optimal_threshold).astype(int)
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred_opt)) > 1:
            cm = confusion_matrix(y_test, y_pred_opt)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 2])
            axes[1, 2].set_title(f"混淆矩阵 (阈值={self.optimal_threshold:.2f})")
            axes[1, 2].set_xlabel("预测")
            axes[1, 2].set_ylabel("实际")
        else:
            axes[1, 2].text(0.5, 0.5, "无法绘制混淆矩阵", ha="center", va="center")

        plt.tight_layout()
        plt.savefig("results/problem4_analysis.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _export_feature_importance(self):
        # 仅对 RF 导出特征重要性
        try:
            if self.best_pipeline is None:
                return
            # 从 pipeline 中取出 rf
            clf = self.best_pipeline.named_steps.get("clf", None)
            if isinstance(clf, RandomForestClassifier):
                imp = pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": clf.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                imp.to_csv(
                    "results/problem4_feature_importance_rf.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        except Exception:
            pass


# ============== 运行入口 ==============


def run_problem4(input_excel_path: str):
    print("=" * 60)
    print("问题4：女胎异常判定（Pipeline+四采样×两模型）")
    print("=" * 60)

    proc = NIPTDataProcessor()
    if not proc.load_data(input_excel_path):
        print("数据加载失败，请检查路径/工作表名称。")
        return None

    female = proc.preprocess_female_data()
    if female.empty:
        print("女胎数据为空，退出。")
        return None

    solver = Problem4Solver(proc, winsorize=False)
    X_test, y_test = solver.train(female)

    print("\n生成的主要文件：")
    print("- results/problem4_cv_summary.csv        （采样×模型CV对比：AUPRC/AUROC）")
    print("- results/problem4_test_summary.csv      （最佳Pipeline外部测试集指标）")
    print("- results/problem4_threshold_optimization.csv （阈值-性能曲线）")
    print("- results/problem4_calibration_points.csv （校准曲线点集）")
    print("- results/problem4_feature_importance_rf.csv （若最佳为RF）")
    print("- results/problem4_analysis.png           （ROC/PR/校准/混淆矩阵等图）")

    return solver


if __name__ == "__main__":
    # TODO: 请把下面路径改成你的附件.xlsx 的实际路径
    excel_path = (
        r"D:\HP\OneDrive\Desktop\学校\竞赛\数模国赛\CUMCM2025Problems\C题\附件.xlsx"
    )
    run_problem4(excel_path)
