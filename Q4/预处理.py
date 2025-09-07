import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings("ignore")


def preprocess_female_data():
    """
    按照数据处理要求对女胎检测数据进行预处理
    """

    # 1. 读取数据
    print("=== 1. 读取数据 ===")
    df = pd.read_csv(
        r"D:\HP\OneDrive\Desktop\学校\竞赛\数模国赛\CUMCM2025Problems\C题\prenatal-testing\Q4\女胎检测数据.csv"
    )
    print(f"原始数据形状: {df.shape}")

    # 2. 标签构造
    print("\n=== 2. 标签构造 ===")
    # 根据"染色体的非整倍体"列构造标签
    df["abnormal_label"] = 0  # 默认正常
    # 如果该列不为空且不为NaN，则为异常
    df.loc[
        df["染色体的非整倍体"].notna() & (df["染色体的非整倍体"] != ""),
        "abnormal_label",
    ] = 1

    # 统计标签分布
    label_counts = df["abnormal_label"].value_counts()
    print(f"标签分布:")
    print(f"正常样本: {label_counts[0]}")
    print(f"异常样本: {label_counts[1]}")
    print(f"样本不平衡比例: {label_counts[0]/label_counts[1]:.2f}:1")

    # 3. 异常频数统计
    print("\n=== 3. 异常频数统计 ===")
    abnormal_samples = df[df["abnormal_label"] == 1]
    if len(abnormal_samples) > 0:
        abnormal_types = abnormal_samples["染色体的非整倍体"].value_counts()
        print("各类异常出现频率:")
        for abnormal_type, count in abnormal_types.items():
            print(f"  {abnormal_type}: {count} 次")

    # 4. 特征选择
    print("\n=== 4. 特征选择 ===")
    # 根据数据处理要求选择特征
    selected_features = [
        # 染色体Z值
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "X染色体的Z值",
        # 染色体GC含量
        "13号染色体的GC含量",
        "18号染色体的GC含量",
        "21号染色体的GC含量",
        # 测序质量与数量相关
        "原始读段数",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "唯一比对的读段数",
        "被过滤掉读段数的比例",
        "GC含量",
        # 孕妇相关
        "孕妇BMI",
        "年龄",
        "身高",
        "体重",
        # X染色体浓度
        "X染色体浓度",
    ]

    # 检查特征是否存在
    existing_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]

    print(f"选择的特征数量: {len(existing_features)}")
    if missing_features:
        print(f"缺失的特征: {missing_features}")

    # 5. 数据清洗和预处理
    print("\n=== 5. 数据清洗 ===")
    # 创建特征数据集
    X = df[existing_features].copy()
    y = df["abnormal_label"].copy()

    # 处理缺失值
    print(f"处理前缺失值统计:")
    missing_counts = X.isnull().sum()
    for feature, count in missing_counts[missing_counts > 0].items():
        print(f"  {feature}: {count}")

    # 用均值填充数值型特征的缺失值
    numerical_features = X.select_dtypes(include=[np.number]).columns
    for feature in numerical_features:
        if X[feature].isnull().sum() > 0:
            X[feature] = X[feature].fillna(X[feature].mean())

    print(f"处理后缺失值数量: {X.isnull().sum().sum()}")

    # GC含量筛选：仅保留0.4-0.6范围内的数据
    print(f"\n=== GC含量筛选 ===")
    print(f"筛选前数据量: {len(X)}")
    if "GC含量" in X.columns:
        gc_mask = (X["GC含量"] >= 0.4) & (X["GC含量"] <= 0.6)
        X = X[gc_mask]
        y = y[gc_mask]
        print(f"GC含量在0.4-0.6范围内的样本数: {len(X)}")
        print(f"筛选后数据量: {len(X)}")
        print(f"GC含量范围: {X['GC含量'].min():.4f} - {X['GC含量'].max():.4f}")
    else:
        print("警告: 未找到GC含量特征")

    # 6. 特征标准化
    print("\n=== 6. 特征标准化 ===")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    print("特征标准化完成")

    # 7. 创建4个不同的数据集
    print("\n=== 7. 创建平衡数据集 ===")

    # 原始数据集
    datasets = {"original": (X_scaled, y)}

    # 随机上采样
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X_scaled, y)
    datasets["oversampled"] = (X_ros, y_ros)
    print(f"上采样后数据形状: {X_ros.shape}")

    # 随机下采样
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_scaled, y)
    datasets["undersampled"] = (X_rus, y_rus)
    print(f"下采样后数据形状: {X_rus.shape}")

    # SMOTE过采样
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
    datasets["smote"] = (X_smote, y_smote)
    print(f"SMOTE后数据形状: {X_smote.shape}")

    # 8. 保存预处理后的数据
    print("\n=== 8. 保存数据 ===")
    import os

    # 确保输出目录存在
    output_dir = "预处理"
    os.makedirs(output_dir, exist_ok=True)

    for name, (X_data, y_data) in datasets.items():
        # 合并特征和标签
        final_data = pd.concat(
            [
                pd.DataFrame(X_data, columns=existing_features),
                pd.Series(y_data, name="abnormal_label"),
            ],
            axis=1,
        )

        # 保存为CSV文件到预处理文件夹
        filename = f"女胎检测数据_预处理后_{name}.csv"
        filepath = os.path.join(output_dir, filename)
        final_data.to_csv(filepath, index=False, encoding="utf-8")
        print(f"已保存: {filepath}")

        # 打印该数据集的标签分布
        label_dist = pd.Series(y_data).value_counts()
        print(f"  - 正常样本: {label_dist[0]}, 异常样本: {label_dist[1]}")

    # 9. 保存特征名称和标准化器
    feature_info = {"selected_features": existing_features, "scaler": scaler}

    import pickle

    pickle_filepath = os.path.join(output_dir, "女胎数据_预处理信息.pkl")
    with open(pickle_filepath, "wb") as f:
        pickle.dump(feature_info, f)

    print("\n=== 预处理完成 ===")
    print("生成的文件:")
    print("- 预处理/女胎检测数据_预处理后_original.csv (原始数据)")
    print("- 预处理/女胎检测数据_预处理后_oversampled.csv (上采样)")
    print("- 预处理/女胎检测数据_预处理后_undersampled.csv (下采样)")
    print("- 预处理/女胎检测数据_预处理后_smote.csv (SMOTE)")
    print("- 预处理/女胎数据_预处理信息.pkl (预处理参数)")

    return datasets, existing_features, scaler


if __name__ == "__main__":
    datasets, features, scaler = preprocess_female_data()
