import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# 尝试导入SMOTE，如果失败则使用简单的类平衡
try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("警告: 未安装imbalanced-learn库，将使用类权重平衡代替SMOTE")

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建results目录
if not os.path.exists('results'):
    os.makedirs('results')
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
            self.male_data = pd.read_excel(file_path, sheet_name='男胎检测数据')
            self.female_data = pd.read_excel(file_path, sheet_name='女胎检测数据')

            # 重命名列为标准格式
            columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE']

            self.male_data.columns = columns[:len(self.male_data.columns)]
            self.female_data.columns = columns[:len(self.female_data.columns)]

            print(f"成功加载数据: 男胎{len(self.male_data)}条, 女胎{len(self.female_data)}条")
            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def parse_gestational_week(self, week_str):
        """解析孕周格式 (如 '11w+6' -> 11.857)"""
        if pd.isna(week_str) or week_str == '':
            return np.nan

        try:
            week_str = str(week_str).strip()
            if 'w' in week_str:
                parts = week_str.split('w')
                weeks = float(parts[0])
                if '+' in parts[1]:
                    days = float(parts[1].replace('+', ''))
                    return weeks + days / 7.0
                else:
                    return weeks
            else:
                return float(week_str)
        except:
            return np.nan

    def preprocess_female_data(self):
        """预处理女胎数据"""
        if self.female_data is None:
            print("请先加载数据")
            return None

        df = self.female_data.copy()

        # 转换孕周格式
        df['J_week'] = df['J'].apply(self.parse_gestational_week)

        # 质量控制过滤
        original_len = len(df)

        df = df[(df['P'] >= 0.35) & (df['P'] <= 0.65)]
        df = df[df['L'] >= 1000000]
        df = df[df['AA'] <= 0.5]

        # 创建异常标签 (AB列不为空表示异常)
        df['abnormal'] = (~df['AB'].isna()).astype(int)

        # 过滤缺失的核心变量
        df = df.dropna(subset=['K', 'J_week', 'Q', 'R', 'S', 'T'])

        print(f"女胎数据预处理: {original_len} -> {len(df)}条记录")
        print(f"异常样本数: {df['abnormal'].sum()}例 ({df['abnormal'].mean() * 100:.1f}%)")

        return df.reset_index(drop=True)


class Problem4Solver:
    """问题4：女胎异常判定分析"""

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.optimal_threshold = 0.5

    def prepare_female_features(self, df):
        """准备女胎特征变量"""
        features = pd.DataFrame()

        # Z值特征（核心指标）
        features['Z13'] = df['Q']  # 13号染色体Z值
        features['Z18'] = df['R']  # 18号染色体Z值
        features['Z21'] = df['S']  # 21号染色体Z值
        features['ZX'] = df['T']  # X染色体Z值

        # X染色体浓度
        if 'W' in df.columns:
            features['X_concentration'] = df['W'].fillna(0)
        else:
            features['X_concentration'] = 0

        # GC含量特征
        features['GC_13'] = df['X']  # 13号染色体GC含量
        features['GC_18'] = df['Y']  # 18号染色体GC含量
        features['GC_21'] = df['Z']  # 21号染色体GC含量
        features['GC_overall'] = df['P']  # 整体GC含量

        # 读段和质量特征
        features['total_reads'] = np.log10(df['L'])  # 总读段数(log变换)
        features['mapped_ratio'] = df['M']  # 比对比例
        features['dup_ratio'] = df['N']  # 重复比例
        features['unique_reads'] = np.log10(df['O'])  # 唯一比对读段数
        features['filtered_ratio'] = df['AA']  # 过滤比例

        # 临床特征
        features['BMI'] = df['K']  # BMI
        features['gestational_week'] = df['J_week']  # 孕周
        features['age'] = df['C']  # 年龄
        features['height'] = df['D']  # 身高
        features['weight'] = df['E']  # 体重

        # 派生特征
        features['Z_max'] = features[['Z13', 'Z18', 'Z21']].abs().max(axis=1)  # 最大Z值绝对值
        features['Z_sum'] = features[['Z13', 'Z18', 'Z21']].abs().sum(axis=1)  # Z值绝对值和
        features['GC_variance'] = features[['GC_13', 'GC_18', 'GC_21']].var(axis=1)  # GC含量方差
        features['GC_mean'] = features[['GC_13', 'GC_18', 'GC_21']].mean(axis=1)  # GC含量均值

        # 交互特征
        features['BMI_age'] = features['BMI'] * features['age']
        features['Z_BMI'] = features['Z_max'] * features['BMI']
        features['Z_week'] = features['Z_max'] * features['gestational_week']

        # 质量评分
        features['quality_score'] = (
                features['mapped_ratio'] * 0.3 +
                (1 - features['dup_ratio']) * 0.3 +
                (1 - features['filtered_ratio']) * 0.4
        )

        # 传统筛查规则特征
        features['high_risk_Z13'] = (features['Z13'].abs() >= 3.0).astype(int)
        features['high_risk_Z18'] = (features['Z18'].abs() >= 3.0).astype(int)
        features['high_risk_Z21'] = (features['Z21'].abs() >= 3.0).astype(int)
        features['very_high_risk'] = (features['Z_max'] >= 3.5).astype(int)
        features['borderline_risk'] = ((features['Z_max'] >= 2.5) &
                                       (features['Z_max'] < 3.5)).astype(int)

        # GC质量规则
        features['gc_abnormal'] = ((features['GC_overall'] < 0.4) |
                                   (features['GC_overall'] > 0.6)).astype(int)

        # 质量控制规则
        features['low_quality'] = (features['quality_score'] < 0.7).astype(int)

        return features

    def handle_class_imbalance(self, X, y, method='class_weight'):
        """处理类别不平衡"""
        print(f"原始数据分布: 正常 {sum(y == 0)}, 异常 {sum(y == 1)}")

        if SMOTE_AVAILABLE and method == 'smote' and sum(y == 1) > 1:
            # 使用SMOTE过采样
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y == 1) - 1))
                X_balanced, y_balanced = smote.fit_resample(X, y)
                print(f"SMOTE后分布: 正常 {sum(y_balanced == 0)}, 异常 {sum(y_balanced == 1)}")
                return X_balanced, y_balanced
            except:
                print("SMOTE失败，使用类权重平衡")
                return X, y
        else:
            # 使用类别权重平衡
            return X, y

    def train_models(self, df):
        """训练多个分类模型"""
        print("开始训练女胎异常判定模型...")

        # 准备数据
        features = self.prepare_female_features(df)
        target = df['abnormal'].values

        print(f"特征数量: {len(features.columns)}")
        print(f"样本数量: {len(target)}")
        print(f"异常比例: {target.mean():.3f}")

        if target.sum() == 0:
            print("警告：没有异常样本，无法训练分类模型")
            return None, None

        # 检查是否有足够的样本进行分层划分
        if target.sum() >= 2 and (1 - target).sum() >= 2:
            stratify = target
        else:
            stratify = None
            print("样本量不足，不使用分层划分")

        # 划分训练测试集
        test_size = min(0.3, max(0.1, 1.0 / len(target)))  # 动态调整测试集比例
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, stratify=stratify
        )

        # 标准化特征
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.feature_names = features.columns.tolist()

        # 处理类别不平衡
        balance_method = 'smote' if SMOTE_AVAILABLE and sum(y_train == 1) > 5 else 'class_weight'
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train_scaled, y_train, balance_method)

        # 1. L1正则化逻辑回归
        print("训练L1正则化逻辑回归...")
        logistic_l1 = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            random_state=42,
            max_iter=1000,
            class_weight='balanced' if balance_method != 'smote' else None
        )
        logistic_l1.fit(X_train_balanced, y_train_balanced)

        # 2. 随机森林
        print("训练随机森林...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        rf_model.fit(X_train_balanced, y_train_balanced)

        # 3. 校准的逻辑回归 (如果样本足够)
        if len(X_train_balanced) >= 20 and sum(y_train_balanced == 1) >= 5:
            print("训练校准逻辑回归...")
            try:
                calibrated_lr = CalibratedClassifierCV(
                    LogisticRegression(C=1.0, random_state=42, max_iter=1000),
                    method='isotonic',
                    cv=min(3, sum(y_train_balanced == 1))
                )
                calibrated_lr.fit(X_train_balanced, y_train_balanced)

                self.models = {
                    'logistic_l1': logistic_l1,
                    'random_forest': rf_model,
                    'calibrated_lr': calibrated_lr
                }
            except:
                print("校准回归训练失败，跳过")
                self.models = {
                    'logistic_l1': logistic_l1,
                    'random_forest': rf_model
                }
        else:
            self.models = {
                'logistic_l1': logistic_l1,
                'random_forest': rf_model
            }

        # 评估模型
        self.evaluate_models(X_test_scaled, y_test)

        return X_test_scaled, y_test

    def evaluate_models(self, X_test, y_test):
        """评估模型性能"""
        print("\n=== 模型性能评估 ===")

        results = []

        for name, model in self.models.items():
            # 预测
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # 计算指标
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_pred_proba)

                # 简化的交叉验证
                if len(X_test) >= 10:
                    try:
                        cv_scores = cross_val_score(model, X_test, y_test, cv=min(3, len(X_test) // 3),
                                                    scoring='roc_auc')
                        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
                    except:
                        cv_mean, cv_std = auc, 0
                else:
                    cv_mean, cv_std = auc, 0
            else:
                auc = 0.5
                cv_mean, cv_std = 0.5, 0

            accuracy = (y_pred == y_test).mean()

            results.append({
                'model': name,
                'auc': auc,
                'accuracy': accuracy,
                'cv_auc_mean': cv_mean,
                'cv_auc_std': cv_std
            })

            print(f"{name}:")
            print(f"  AUC: {auc:.3f}")
            print(f"  准确率: {accuracy:.3f}")
            print(f"  CV AUC: {cv_mean:.3f} ± {cv_std:.3f}")
            print()

        self.evaluation_results = pd.DataFrame(results)
        return self.evaluation_results

    def optimize_threshold(self, X_test, y_test, cost_ratio=10):
        """优化决策阈值"""
        print("优化决策阈值...")

        if len(self.models) == 0 or len(np.unique(y_test)) == 1:
            self.optimal_threshold = 0.5
            return 0.5

        # 使用最佳模型
        best_model_name = self.evaluation_results.loc[
            self.evaluation_results['auc'].idxmax(), 'model'
        ]
        best_model = self.models[best_model_name]

        y_proba = best_model.predict_proba(X_test)[:, 1]

        # 计算不同阈值下的性能
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = -np.inf

        threshold_results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
                # 混淆矩阵
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()

                        # 性能指标
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

                        # 成本敏感评分 (假阴性成本更高)
                        cost_score = sensitivity * 0.7 + specificity * 0.3 - cost_ratio * fn / len(y_test)

                        threshold_results.append({
                            'threshold': threshold,
                            'sensitivity': sensitivity,
                            'specificity': specificity,
                            'precision': precision,
                            'cost_score': cost_score,
                            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                        })

                        if cost_score > best_score:
                            best_score = cost_score
                            best_threshold = threshold
                except:
                    continue

        self.optimal_threshold = best_threshold
        if threshold_results:
            self.threshold_results = pd.DataFrame(threshold_results)
            best_result = max(threshold_results, key=lambda x: x['cost_score'])
            print(f"最优阈值: {best_threshold:.3f}")
            print(f"对应性能: 敏感性 {best_result['sensitivity']:.3f}, "
                  f"特异性 {best_result['specificity']:.3f}")
        else:
            print(f"使用默认阈值: {best_threshold:.3f}")

        return best_threshold

    def create_decision_rules(self, df, X_test, y_test):
        """创建决策规则"""
        print("创建女胎异常判定规则...")

        if len(self.models) == 0:
            return {}

        # 获取最佳模型
        best_model_name = self.evaluation_results.loc[
            self.evaluation_results['auc'].idxmax(), 'model'
        ]
        best_model = self.models[best_model_name]

        # 预测概率
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # 规则框架
        rules = {
            'direct_abnormal': [],  # 直接判为异常
            'high_risk_retest': [],  # 高风险复检
            'model_based': [],  # 基于模型
            'normal': []  # 正常
        }

        # 基于Z值的直接规则
        test_indices = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
        features_test = self.prepare_female_features(df.iloc[test_indices])

        for i, (idx, row) in enumerate(features_test.iterrows()):
            z_max = max(abs(row['Z13']), abs(row['Z18']), abs(row['Z21']))
            prob = y_proba[i] if i < len(y_proba) else 0.5

            if z_max >= 3.5:
                rules['direct_abnormal'].append(i)
            elif 2.5 <= z_max < 3.5 and 0.4 <= prob <= 0.6:
                rules['high_risk_retest'].append(i)
            elif prob >= self.optimal_threshold:
                rules['model_based'].append(i)
            else:
                rules['normal'].append(i)

        # 统计规则效果
        rule_stats = {}
        for rule_name, indices in rules.items():
            if len(indices) > 0:
                if hasattr(y_test, 'iloc'):
                    actual_abnormal = y_test.iloc[indices].sum()
                else:
                    actual_abnormal = sum(y_test[indices])

                rule_stats[rule_name] = {
                    'count': len(indices),
                    'abnormal_count': actual_abnormal,
                    'precision': actual_abnormal / len(indices) if len(indices) > 0 else 0
                }

        print("决策规则统计:")
        for rule_name, stats in rule_stats.items():
            print(f"{rule_name}: {stats['count']}例, 阳性率 {stats['precision']:.3f}")

        self.decision_rules = rules
        self.rule_stats = rule_stats

        return rules

    def analyze_feature_importance(self):
        """分析特征重要性"""
        importance_dfs = []

        # 使用随机森林的特征重要性
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_dfs.append(('random_forest', importance_df))

        # 使用L1逻辑回归的系数
        if 'logistic_l1' in self.models:
            lr_model = self.models['logistic_l1']
            coef_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': lr_model.coef_[0],
                'abs_coefficient': np.abs(lr_model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            importance_dfs.append(('logistic_l1', coef_df))

        if importance_dfs:
            print("特征重要性分析:")
            for model_name, df in importance_dfs:
                print(f"\n{model_name} Top 10:")
                if 'importance' in df.columns:
                    print(df[['feature', 'importance']].head(10).to_string(index=False))
                else:
                    print(df[['feature', 'abs_coefficient']].head(10).to_string(index=False))

        return importance_dfs

    def plot_results(self, df, X_test, y_test):
        """绘制分析结果"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('女胎异常判定模型分析结果', fontsize=16)

        # 检查是否有有效的测试数据和模型
        if X_test is None or y_test is None or len(self.models) == 0:
            axes[0, 0].text(0.5, 0.5, '模型训练失败或数据不足', ha='center', va='center')
            plt.tight_layout()
            return fig

        # 使用最佳模型进行预测
        best_model_name = self.evaluation_results.loc[self.evaluation_results['auc'].idxmax(), 'model']
        best_model = self.models[best_model_name]
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # 1. ROC曲线
        if len(np.unique(y_test)) > 1:
            for name, model in self.models.items():
                y_proba_model = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba_model)
                auc = roc_auc_score(y_test, y_proba_model)
                axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

            axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC曲线')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, '无法绘制ROC曲线\n(单一类别)', ha='center', va='center')

        # 2. Precision-Recall曲线
        if len(np.unique(y_test)) > 1:
            for name, model in self.models.items():
                y_proba_model = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_proba_model)
                axes[0, 1].plot(recall, precision, label=name)

            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall曲线')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '无法绘制PR曲线\n(单一类别)', ha='center', va='center')

        # 3. 校准曲线
        if len(np.unique(y_test)) > 1 and len(y_proba) > 10:
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_proba, n_bins=min(5, len(y_test) // 2))
                axes[0, 2].plot(mean_predicted_value, fraction_of_positives, "s-", label=best_model_name)
                axes[0, 2].plot([0, 1], [0, 1], "k:", label="完全校准")
                axes[0, 2].set_xlabel('平均预测概率')
                axes[0, 2].set_ylabel('实际阳性率')
                axes[0, 2].set_title('校准曲线')
                axes[0, 2].legend()
            except:
                axes[0, 2].text(0.5, 0.5, '校准曲线绘制失败', ha='center', va='center')
        else:
            axes[0, 2].text(0.5, 0.5, '数据不足以绘制校准曲线', ha='center', va='center')

        # 4. 特征重要性
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importances = rf_model.feature_importances_
            top_indices = np.argsort(importances)[-min(15, len(importances)):]

            axes[1, 0].barh(range(len(top_indices)), importances[top_indices])
            axes[1, 0].set_yticks(range(len(top_indices)))
            axes[1, 0].set_yticklabels([self.feature_names[i] for i in top_indices])
            axes[1, 0].set_xlabel('重要性')
            axes[1, 0].set_title('特征重要性 (随机森林)')

        # 5. Z值分布
        features = self.prepare_female_features(df)
        z_columns = ['Z13', 'Z18', 'Z21']
        for col in z_columns:
            if col in features.columns:
                axes[1, 1].hist(features[col], alpha=0.5, label=col, bins=20)

        axes[1, 1].axvline(3.0, color='red', linestyle='--', alpha=0.7, label='传统阈值(+)')
        axes[1, 1].axvline(-3.0, color='red', linestyle='--', alpha=0.7, label='传统阈值(-)')
        axes[1, 1].set_xlabel('Z值')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('Z值分布')
        axes[1, 1].legend()

        # 6. 阈值优化结果
        if hasattr(self, 'threshold_results') and len(self.threshold_results) > 0:
            thresh_df = self.threshold_results
            axes[1, 2].plot(thresh_df['threshold'], thresh_df['sensitivity'], 'o-', label='敏感性')
            axes[1, 2].plot(thresh_df['threshold'], thresh_df['specificity'], 's-', label='特异性')
            axes[1, 2].axvline(self.optimal_threshold, color='red', linestyle='--', label='最优阈值')
            axes[1, 2].set_xlabel('阈值')
            axes[1, 2].set_ylabel('性能指标')
            axes[1, 2].set_title('阈值优化')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, '阈值优化结果不可用', ha='center', va='center')

        # 7. 混淆矩阵
        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
            axes[2, 0].set_xlabel('预测')
            axes[2, 0].set_ylabel('实际')
            axes[2, 0].set_title(f'混淆矩阵 (阈值={self.optimal_threshold:.3f})')
        else:
            axes[2, 0].text(0.5, 0.5, '无法生成混淆矩阵', ha='center', va='center')

        # 8. 决策规则分布
        if hasattr(self, 'rule_stats') and self.rule_stats:
            rule_names = list(self.rule_stats.keys())
            rule_counts = [self.rule_stats[name]['count'] for name in rule_names]

            axes[2, 1].bar(range(len(rule_names)), rule_counts, alpha=0.7)
            axes[2, 1].set_xticks(range(len(rule_names)))
            axes[2, 1].set_xticklabels([name.replace('_', '\n') for name in rule_names], rotation=0)
            axes[2, 1].set_ylabel('样本数')
            axes[2, 1].set_title('决策规则分布')
        else:
            axes[2, 1].text(0.5, 0.5, '决策规则统计不可用', ha='center', va='center')

        # 9. 模型性能对比
        if hasattr(self, 'evaluation_results') and len(self.evaluation_results) > 0:
            model_names = self.evaluation_results['model']
            aucs = self.evaluation_results['auc']

            bars = axes[2, 2].bar(range(len(model_names)), aucs, alpha=0.7)
            axes[2, 2].set_xticks(range(len(model_names)))
            axes[2, 2].set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=0)
            axes[2, 2].set_ylabel('AUC')
            axes[2, 2].set_title('模型性能对比')
            axes[2, 2].set_ylim([0.4, 1.0])

            # 标注数值
            for bar, auc in zip(bars, aucs):
                axes[2, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f'{auc:.3f}', ha='center', va='bottom')
        else:
            axes[2, 2].text(0.5, 0.5, '模型性能对比不可用', ha='center', va='center')

        plt.tight_layout()
        return fig

    def generate_prediction_report(self, df, X_test, y_test):
        """生成预测报告"""
        if X_test is None or y_test is None or len(self.models) == 0:
            return {}, {}

        best_model_name = self.evaluation_results.loc[self.evaluation_results['auc'].idxmax(), 'model']
        best_model = self.models[best_model_name]

        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.optimal_threshold).astype(int)

        # 详细分类报告
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
            try:
                report = classification_report(y_test, y_pred, output_dict=True)
            except:
                report = {}
        else:
            report = {}

        # 高风险样本分析
        high_risk_mask = y_proba >= 0.8
        high_risk_indices = np.where(high_risk_mask)[0]
        high_risk_actual = y_test[high_risk_indices] if len(high_risk_indices) > 0 else []

        summary = {
            'model_used': best_model_name,
            'optimal_threshold': self.optimal_threshold,
            'total_samples': len(y_test),
            'predicted_abnormal': int(sum(y_pred)),
            'actual_abnormal': int(sum(y_test)),
            'high_risk_samples': len(high_risk_indices),
            'high_risk_accuracy': sum(high_risk_actual) / len(high_risk_indices) if len(high_risk_indices) > 0 else 0,
            'overall_accuracy': (y_pred == y_test).mean() if len(y_test) > 0 else 0,
            'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
        }

        # 添加分类报告中的指标
        if report and '1' in report:
            summary.update({
                'precision': report['1'].get('precision', 0),
                'recall': report['1'].get('recall', 0),
                'f1_score': report['1'].get('f1-score', 0)
            })
        else:
            summary.update({
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            })

        return summary, report


def run_problem4():
    """运行问题4分析"""
    print("=" * 60)
    print("问题4：女胎异常判定分析")
    print("=" * 60)

    # 初始化数据处理器
    processor = NIPTDataProcessor()

    # 加载数据
    if not processor.load_data('附件.xlsx'):
        print("数据加载失败，请检查文件'附件.xlsx'是否存在")
        return None

    female_df = processor.preprocess_female_data()
    if female_df is None or len(female_df) == 0:
        print("女胎数据预处理失败")
        return None

    print(f"女胎数据: {len(female_df)}例")
    print(f"异常病例: {sum(~female_df['AB'].isna())}例")

    # 创建求解器
    solver = Problem4Solver(processor)

    # 步骤1：训练模型
    print("\n步骤1：训练女胎异常判定模型")
    X_test, y_test = solver.train_models(female_df)

    if X_test is not None and y_test is not None:
        # 步骤2：优化阈值
        print("\n步骤2：优化决策阈值")
        optimal_threshold = solver.optimize_threshold(X_test, y_test)

        # 步骤3：创建决策规则
        print("\n步骤3：创建决策规则")
        decision_rules = solver.create_decision_rules(female_df, X_test, y_test)

        # 步骤4：分析特征重要性
        print("\n步骤4：分析特征重要性")
        importance_results = solver.analyze_feature_importance()

        # 步骤5：生成预测报告
        print("\n步骤5：生成预测报告")
        summary, detailed_report = solver.generate_prediction_report(female_df, X_test, y_test)

        # 步骤6：绘制结果
        print("\n步骤6：生成可视化结果")
        fig = solver.plot_results(female_df, X_test, y_test)
        plt.savefig('results/problem4_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 步骤7：保存结果
        print("\n步骤7：保存分析结果")

        if summary:
            # 保存预测报告摘要
            pd.DataFrame([summary]).to_csv('results/problem4_summary.csv', index=False, encoding='utf-8-sig')

            # 保存特征重要性
            if importance_results:
                for model_name, importance_df in importance_results:
                    importance_df.to_csv(f'results/problem4_feature_importance_{model_name}.csv',
                                         index=False, encoding='utf-8-sig')

            # 保存模型评估结果
            if hasattr(solver, 'evaluation_results'):
                solver.evaluation_results.to_csv('results/problem4_model_comparison.csv',
                                                 index=False, encoding='utf-8-sig')

            # 保存阈值优化结果
            if hasattr(solver, 'threshold_results'):
                solver.threshold_results.to_csv('results/problem4_threshold_optimization.csv',
                                                index=False, encoding='utf-8-sig')

            # 保存决策规则统计
            if hasattr(solver, 'rule_stats'):
                rule_stats_df = pd.DataFrame([
                    {'rule': rule_name, **stats}
                    for rule_name, stats in solver.rule_stats.items()
                ])
                rule_stats_df.to_csv('results/problem4_decision_rules.csv',
                                     index=False, encoding='utf-8-sig')

            print("=" * 60)
            print("问题4分析完成！")
            print("生成的文件：")
            print("- results/problem4_analysis.png (可视化结果)")
            print("- results/problem4_summary.csv (预测报告摘要)")
            print("- results/problem4_model_comparison.csv (模型性能对比)")
            print("- results/problem4_feature_importance_*.csv (特征重要性)")
            print("- results/problem4_threshold_optimization.csv (阈值优化)")
            print("- results/problem4_decision_rules.csv (决策规则统计)")
            print("=" * 60)

            # 输出关键结果摘要
            print("\n=== 关键结果摘要 ===")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

            # 决策规则建议
            print("\n=== 女胎异常判定建议 ===")
            print("1. 直接异常判定: Z值最大绝对值 ≥ 3.5")
            print("2. 高风险复检: 2.5 ≤ Z值最大绝对值 < 3.5 且模型预测概率在0.4-0.6之间")
            print(f"3. 模型判定: 模型预测概率 ≥ {solver.optimal_threshold:.3f}")
            print("4. 正常: 其他情况")

            print(f"\n最佳模型: {summary['model_used']}")
            print(f"整体准确率: {summary['overall_accuracy']:.3f}")
            print(f"AUC: {summary['auc']:.3f}")

            if hasattr(solver, 'rule_stats'):
                print("\n各决策规则性能:")
                for rule_name, stats in solver.rule_stats.items():
                    print(f"- {rule_name}: {stats['count']}例, 阳性率 {stats['precision']:.3f}")

        else:
            print("警告：未能生成有效的预测报告")

    else:
        print("警告：模型训练失败，无法进行后续分析")

    return solver


if __name__ == "__main__":
    run_problem4()