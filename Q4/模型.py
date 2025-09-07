# 模型.py
# 实现第四问：数据 → 建模 → 评估 → 汇报

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
)  # 使用 scikit-learn 的 GBM 实现，避免外部依赖
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def load_preprocessed(pre_dir="预处理"):
    """加载预处理后四种数据集"""
    datasets = {}
    for name in ["original", "oversampled", "undersampled", "smote"]:
        path = os.path.join(pre_dir, f"女胎检测数据_预处理后_{name}.csv")
        df = pd.read_csv(path)
        X = df.drop("abnormal_label", axis=1)
        y = df["abnormal_label"]
        datasets[name] = (X, y)
    return datasets


def data_profile(raw_csv):
    """统计原始异常类型频次"""
    df = pd.read_csv(raw_csv)
    df["abnormal_label"] = 0
    df.loc[
        df["染色体的非整倍体"].notna() & (df["染色体的非整倍体"] != ""),
        "abnormal_label",
    ] = 1
    counts = df["染色体的非整倍体"].value_counts()
    print("异常类型频次:")
    print(counts)
    print()


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
    }


def main():
    # 1. 加载预处理数据并侧写原始异常
    datasets = load_preprocessed()
    print("== 数据侧写 ==")
    raw_csv = os.path.join(os.getcwd(), "女胎检测数据.csv")
    data_profile(raw_csv)

    # 2. 划分外部测试集
    X_orig, y_orig = datasets["original"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42
    )

    # 3. 定义采样策略与模型及参数
    samplers = {
        "original": None,
        "oversampled": RandomOverSampler(random_state=42),
        "undersampled": RandomUnderSampler(random_state=42),
        "smote": SMOTE(random_state=42),
    }
    models = {
        "GBM": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
            },
        )
    }

    # 4. 交叉验证调参并测试评估
    results = []
    best_models = {}
    for s_name, sampler in samplers.items():
        if sampler:
            X_tr, y_tr = sampler.fit_resample(X_train, y_train)
        else:
            X_tr, y_tr = X_train, y_train
        for m_name, (mdl, params) in models.items():
            print(f"训练: {s_name} - {m_name}")
            grid = GridSearchCV(mdl, params, cv=5, scoring="roc_auc", n_jobs=-1)
            grid.fit(X_tr, y_tr)
            best = grid.best_estimator_
            best_models[(s_name, m_name)] = best
            met = evaluate(best, X_test, y_test)
            met.update({"sampler": s_name, "model": m_name})
            results.append(met)

    # 5. 汇总与可视化
    df_res = pd.DataFrame(results)
    print("\n== 测试集结果汇总 ==")
    print(df_res)
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/model_comparison.csv", index=False)

    # 平行坐标图
    pd.plotting.parallel_coordinates(
        df_res, "sampler", cols=["roc_auc", "average_precision", "f1"]
    )
    plt.title("采样策略与模型对比")
    plt.savefig("results/parallel_coordinates.png")
    plt.close()

    # 绘制最优模型的 ROC 和 PR 曲线
    best_row = df_res.loc[df_res["roc_auc"].idxmax()]
    bs, bm = best_row["sampler"], best_row["model"]
    best_est = best_models[(bs, bm)]
    y_prob = best_est.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC ({bm})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig("results/roc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"PR ({bm})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("results/pr_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
