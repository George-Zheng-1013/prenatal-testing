import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations
import statsmodels.api as sm
from lifelines.statistics import logrank_test  # 新增导入

try:
    # 部分类型检查器可能无法识别 patsy 顶层导出，添加忽略标注
    from patsy import dmatrix  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from patsy.highlevel import dmatrix  # 备用导入

THRESHOLD = 0.04  # Y染色体浓度达标阈值
BOOT_N = 100  # bootstrap 次数
RANDOM_STATE = 42
TARGET = 0.95  # 全局目标达标率
# 孕周窗口与偏好（越早越好）
EARLY_END = 12.0
MID_END = 27.0
LATE_START = 28.0

# 在常量区新增一个开关
GROUPING_METHOD = "supervised"  # 可选: "quantile"(现有逻辑), "clinical", "supervised"
MIN_GROUP_SIZE = 30  # 监督式分组每个子组的最小样本量
MAX_GROUPS = 6  # 监督式分组最多分组数
# 新增：分段阶段的目标达标率（比最终报告略低，提升可分性）
SEG_TARGET = 0.90


@dataclass
class GroupResult:
    group_label: str
    group_index: int
    n: int
    censor_rate: float
    t_star: Optional[float]
    t_star_ci_low: Optional[float]
    t_star_ci_high: Optional[float]
    F_at_t_star: Optional[float]
    # 早/中/晚窗口信息与关键孕周达标率
    t_star_window: str
    F_at_12w: Optional[float]
    F_at_13w: Optional[float]
    F_at_28w: Optional[float]
    early_alt_t90: Optional[float]


def preprocess_data(
    csv_path: str, threshold: float = THRESHOLD
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取数据，筛选男胎个体，构建个体层面的首次达标事件/删失数据。

    返回：
    - df_obs_male: 观测层数据（仅男胎个体的所有检测记录），包含列：孕妇代码, 检测孕周, 孕妇BMI, Y染色体浓度, 达标(bool)
    - df_subjects: 个体层数据（每位孕妇一行），包含列：孕妇代码, 孕妇BMI, duration_week, event
    """
    # 处理中文路径 & 编码
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = ["孕妇代码", "检测孕周", "孕妇BMI", "Y染色体浓度"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 只保留必要列并清洗
    df = df[required_cols].copy()
    # 类型转换
    for c in ["检测孕周", "孕妇BMI", "Y染色体浓度"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["孕妇代码", "检测孕周", "孕妇BMI", "Y染色体浓度"]).copy()

    # 标注达标
    df["达标"] = df["Y染色体浓度"] >= threshold

    df_male = df.copy()

    # 个体层：首次达标孕周 / 若未达标则以最后一次孕周作为删失时间
    def _first_hit_or_censor(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("检测孕周")
        hits = g[g["达标"]]
        if len(hits) > 0:
            t = float(hits["检测孕周"].iloc[0])
            event = True
        else:
            t = float(g["检测孕周"].max())
            event = False
        bmi = float(g["孕妇BMI"].iloc[0])
        return pd.Series({"duration_week": t, "event": event, "孕妇BMI": bmi})

    df_subjects = (
        df_male.groupby("孕妇代码", as_index=False)[["检测孕周", "达标", "孕妇BMI"]]
        .apply(_first_hit_or_censor)
        .reset_index(drop=True)
    )

    return df_male.reset_index(drop=True), df_subjects


def group_bmi(
    df_subjects: pd.DataFrame, n_groups: int = 5
) -> Tuple[pd.DataFrame, List[float]]:
    """按BMI五分位数分组，并返回组边界。

    返回：
    - df_subjects_g: 新增列 'bmi_group' (1..k) 与 'bmi_group_label'
    - boundaries: 分位数边界值列表，长度 n_groups+1
    """
    bmi = df_subjects["孕妇BMI"].astype(float)

    # 计算分位数边界
    qs = np.linspace(0, 1, n_groups + 1)
    boundaries = [float(np.quantile(bmi, q)) for q in qs]
    # 处理重复边界（当BMI重复较多时）
    boundaries = np.unique(np.round(boundaries, 6)).tolist()
    if len(boundaries) - 1 < n_groups:
        # 若唯一边界不足，退回到qcut(duplicates='drop')并读取其实际边界
        cat = pd.qcut(bmi, q=n_groups, duplicates="drop")
        categories = cat.cat.categories
        # 从Interval中萃取边界
        edges = [categories[0].left] + [iv.right for iv in categories]
        boundaries = [float(e) for e in edges]

    # 用 qcut 获取组别（尽量等量分组）
    cat = pd.qcut(bmi, q=min(n_groups, len(boundaries) - 1), duplicates="drop")
    # 将区间转为 1..k 序号，并生成可读标签
    codes = cat.cat.codes + 1
    labels = [
        f"Q{c}: [{iv.left:.2f}, {iv.right:.2f})"
        for c, iv in zip(codes, cat.cat.categories.take(cat.cat.codes))
    ]

    df_subjects_g = df_subjects.copy()
    df_subjects_g["bmi_group"] = codes.values
    df_subjects_g["bmi_group_label"] = labels

    return df_subjects_g, boundaries


def km_estimate_t_star(
    durations: np.ndarray, events: np.ndarray, target: float = TARGET
) -> Tuple[Optional[float], Optional[float]]:
    """根据KM估计累计达标率F(t)=1-S(t)，返回最早 t* 使 F(t)≥target 以及 F(t*)。
    若未达到target，返回达到的最大F与对应t（按事件时间节点）。
    返回 (t_star, F_at_tStar)。若未达到target则 t_star为能达到最高比例的时刻。
    """
    if len(durations) == 0:
        return None, None

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    # survival_function_ 是 S(t)，我们要 F(t)=1-S(t)
    sf = kmf.survival_function_.copy()
    sf.columns = ["S"]
    sf["F"] = 1 - sf["S"]

    # 在事件时刻寻找最早达到 target 的时间
    reached = sf[sf["F"] >= target]
    if len(reached) > 0:
        t_star = float(reached.index[0])
        F_at = float(reached.iloc[0]["F"])
        return t_star, F_at
    else:
        # 未达到 target，返回最大值
        t_star = float(sf.index.max())
        F_at = float(sf["F"].iloc[-1])
        return t_star, F_at


def bootstrap_t_star(
    durations: np.ndarray,
    events: np.ndarray,
    n_boot: int = BOOT_N,
    target: float
    """对个体层数据进行bootstrap，返回 t* 的 95% CI（2.5%, 97.5%）。
    若大多数样本未达到 target，可能返回 (None, None)。
    """
    rng = np.random.RandomState(random_state)
    n = len(durations)
    if n == 0:
        return None, None

    tstars: List[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        d = durations[idx]
        e = events[idx]
        t_star, F_at = km_estimate_t_star(d, e, target=target)
        # 仅记录真正达到 target 的 t*，否则视为缺失
        if t_star is not None and F_at is not None and F_at >= target:
            tstars.append(t_star)

    if len(tstars) < max(3, n_boot // 5):
        # 过少可用样本，不给出区间
        return None, None

    low, high = np.percentile(tstars, [2.5, 97.5])
    return float(low), float(high)


def classify_window(t: Optional[float]) -> str:
    """根据孕周 t 给出窗口标签。"""
    if t is None:
        return "Not reached"
    if t <= EARLY_END:
        return "Early (<=12w)"
    if t < LATE_START:
        return "Mid (13-27w)"
    return "Late (>=28w)"


def earliest_t_within_window(
    sf: pd.DataFrame, target: float, t_max: float
) -> Optional[float]:
    """在KM累计达标率曲线 F(t) 中，寻找 t<=t_max 的最早时刻使 F(t)>=target。若不存在返回None。
    sf 需包含列 'F'，索引为时间点（事件时刻）。"""
    sub = sf[sf.index <= t_max]
    if sub.empty:
        return None
    reached = sub[sub["F"] >= target]
    if len(reached) > 0:
        return float(reached.index[0])
    return None


def fit_logistic_smooth(df_obs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """用观测层数据(每次抽血记录)的二分类达标 ~ 孕周，拟合GLM-Binomial并返回平滑曲线。
    使用样条基函数以获得平滑曲线。
    返回：grid_t(孕周网格), pred_prob(预测达标概率)
    """
    df_obs = df_obs.dropna(subset=["检测孕周", "达标"]).copy()
    if df_obs.empty:
        return np.array([]), np.array([])

    # 构建设计矩阵（自然三次样条）
    # 确保为 numpy float 数组
    x = pd.to_numeric(df_obs["检测孕周"], errors="coerce").to_numpy(dtype=float)
    X = dmatrix(
        "bs(x, df=4, include_intercept=True)", {"x": x}, return_type="dataframe"
    )
    y = df_obs["达标"].astype(int).values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X, family=sm.families.Binomial())
        res = model.fit()

    grid_t = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    Xg = dmatrix(
        "bs(x, df=4, include_intercept=True)", {"x": grid_t}, return_type="dataframe"
    )
    pred = res.predict(Xg)
    pred = np.clip(pred, 0, 1)
    return grid_t, pred


def analyze_group(
    group_idx: int,
    group_label: str,
    df_obs_group: pd.DataFrame,
    df_subj_group: pd.DataFrame,
    out_dir: str,
) -> GroupResult:
    """对单个BMI组进行分析与绘图，返回汇总结果。"""
    durations = df_subj_group["duration_week"].values.astype(float)
    events = df_subj_group["event"].values.astype(bool)

    # KM估计与 t*
    t_star, F_at_t = km_estimate_t_star(durations, events, target=TARGET)
    ci_low, ci_high = bootstrap_t_star(durations, events, n_boot=BOOT_N, target=TARGET)

    # 删失比例
    n = len(df_subj_group)
    censor_rate = float(1 - events.mean()) if n > 0 else np.nan

    # 拟合logistic平滑
    grid_t, pred_prob = fit_logistic_smooth(df_obs_group)

    # KM 曲线数据（转换为累计达标率）+ 置信带
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    sf = kmf.survival_function_.copy()
    sf.columns = ["S"]
    sf["F"] = 1 - sf["S"]
    ci = kmf.confidence_interval_.copy()
    # lifelines置信区间列名通常为 'KM_estimate_lower_0.95'/'KM_estimate_upper_0.95'
    lower_col = [c for c in ci.columns if "lower" in c][0]
    upper_col = [c for c in ci.columns if "upper" in c][0]
    F_lower = 1 - ci[upper_col]
    F_upper = 1 - ci[lower_col]

    # 关键孕周点的达标率（使用S(t)在给定时刻的估计）
    try:
        S_at = kmf.survival_function_at_times([EARLY_END, 13.0, LATE_START])
        # survival_function_at_times 返回Series，索引为给定时刻
        F_at_12w = float(1 - S_at.loc[EARLY_END])
        F_at_13w = float(1 - S_at.loc[13.0])
        F_at_28w = float(1 - S_at.loc[LATE_START])
    except Exception:
        # 兜底（极端情况下）
        F_at_12w = (
            float(sf["F"].loc[sf.index[sf.index <= EARLY_END]].iloc[-1])
            if (sf.index <= EARLY_END).any()
            else float(sf["F"].iloc[0])
        )
        F_at_13w = (
            float(sf["F"].loc[sf.index[sf.index <= 13.0]].iloc[-1])
            if (sf.index <= 13.0).any()
            else float(sf["F"].iloc[0])
        )
        F_at_28w = (
            float(sf["F"].loc[sf.index[sf.index <= LATE_START]].iloc[-1])
            if (sf.index <= LATE_START).any()
            else float(sf["F"].iloc[-1])
        )

    # t* 所属窗口
    t_star_window = classify_window(t_star)

    # 绘图
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    # 背景着色：早/中/晚
    xmax = float(max(sf.index.max(), LATE_START + 1))
    ax.axvspan(0, EARLY_END, color="#2ca02c", alpha=0.08, label="Early (<=12w)")
    ax.axvspan(EARLY_END, MID_END, color="#ff7f0e", alpha=0.06, label="Mid (13-27w)")
    ax.axvspan(LATE_START, xmax, color="#d62728", alpha=0.05, label="Late (>=28w)")
    # KM累计达标率
    ax.step(
        sf.index,
        sf["F"],
        where="post",
        label="KM cumulative success rate",
        color="#1f77b4",
    )
    ax.fill_between(
        sf.index,
        F_lower,
        F_upper,
        step="post",
        color="#1f77b4",
        alpha=0.2,
        label="KM 95% CI",
    )

    # Logistic 平滑
    if len(grid_t) > 0:
        ax.plot(
            grid_t,
            pred_prob,
            color="#d62728",
            linestyle="--",
            label="Logistic smoothing",
        )

    # 标注 t*
    if t_star is not None:
        ax.axvline(float(t_star), color="#2ca02c", linestyle=":", alpha=0.8)
        ax.scatter(
            [float(t_star)],
            [float(F_at_t) if F_at_t is not None else np.nan],
            color="#2ca02c",
            zorder=3,
        )
        ax.annotate(
            f"Recommended GA t*={float(t_star):.2f}\nF(t*)={(float(F_at_t) if F_at_t is not None else np.nan):.2%}",
            xy=(float(t_star), float(F_at_t) if F_at_t is not None else np.nan),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="#2ca02c",
        )

    ax.set_title(f"BMI group {group_label}")
    ax.set_xlabel("Gestational age (weeks)")
    ax.set_ylabel("Cumulative success rate F(t)")
    ax.set_ylim(0, 1)
    # 去重图例项并保持顺序
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            new_handles.append(h)
            new_labels.append(l)
    ax.legend(new_handles, new_labels, loc="lower right")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"km_bmi_group_{group_idx}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    return GroupResult(
        group_label=group_label,
        group_index=group_idx,
        n=n,
        censor_rate=censor_rate,
        t_star=float(t_star) if t_star is not None else None,
        t_star_ci_low=float(ci_low) if ci_low is not None else None,
        t_star_ci_high=float(ci_high) if ci_high is not None else None,
        F_at_t_star=float(F_at_t) if F_at_t is not None else None,
        t_star_window=t_star_window,
        F_at_12w=float(F_at_12w) if F_at_12w is not None else None,
        F_at_13w=float(F_at_13w) if F_at_13w is not None else None,
        F_at_28w=float(F_at_28w) if F_at_28w is not None else None,
        early_alt_t90=None,  # 早孕窗口内的备选（已移除，不再计算）
    )


def group_bmi_clinical(df_subjects: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
    """按照题目给定的分组"""
    bins = [20, 28, 32, 36, 40, np.inf]
    labels = ["20–28", "28–32", "32–36", "36–40", ">=40"]
    cat = pd.cut(
        df_subjects["孕妇BMI"].astype(float), bins=bins, labels=labels, right=False
    )
    df = df_subjects.copy()
    df["bmi_group"] = cat.cat.codes + 1
    df["bmi_group_label"] = [f"CN {l}" for l in cat.astype(str)]
    # 返回边界列表
    boundaries = [20.0, 28.0, 32.0, 36.0, 40.0, float(1e9)]
    return df, boundaries


def _segment_score(
    durations: np.ndarray, events: np.ndarray, target: float = TARGET
) -> Tuple[float, float]:
    """返回(组内t*, F(12w))，用于评估切分收益。"""
    t_star, F_at = km_estimate_t_star(durations, events, target=target)
    kmf = KaplanMeierFitter().fit(durations, events)
    try:
        F12 = float(1 - kmf.survival_function_at_times([EARLY_END]).iloc[0])
    except Exception:
        sf = kmf.survival_function_.copy()
        sf["F"] = 1 - sf.iloc[:, 0]
        F12 = (
            float(sf["F"].loc[sf.index[sf.index <= EARLY_END]].iloc[-1])
            if (sf.index <= EARLY_END).any()
            else float(sf["F"].iloc[0])
        )
    return (float(t_star) if t_star is not None else np.inf, F12)


def group_bmi_supervised(
    df_subjects: pd.DataFrame,
    max_groups: int = MAX_GROUPS,
    min_size: int = MIN_GROUP_SIZE,
    target: float = TARGET,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    监督式分组（生存树思想）：
    - 在 BMI 上寻找切分点，使加权平均 t* 尽可能小（并兼顾提高 F(12w)）。
    - 迭代二分直到达到 max_groups 或无显著提升。
    """
    df = df_subjects.copy()
    df = df.sort_values("孕妇BMI").reset_index(drop=True)
    segments = [(0, len(df))]  # 用索引区间表示段
    # 预先准备数据
    bmi = df["孕妇BMI"].to_numpy(float)
    dur = df["duration_week"].to_numpy(float)
    evt = df["event"].to_numpy(bool)

    def segment_indices(seg):  # 返回该段内的布尔掩码
        i, j = seg
        mask = np.zeros(len(df), dtype=bool)
        mask[i:j] = True
        return mask

    while len(segments) < max_groups:
        best_gain = 0.0
        best_split = None  # (seg_idx, cut_bmi, left_mask, right_mask)

        # 计算当前总目标函数：加权平均 t*（对每段）
        def current_objective(segs):
            num, den = 0.0, 0.0
            for i, j in segs:
                m = segment_indices((i, j))
                tstar, F12 = _segment_score(dur[m], evt[m], target=target)
                n = m.sum()
                num += tstar * n
                den += n
            return num / max(den, 1.0)

        base_obj = current_objective(segments)

        for si, (i, j) in enumerate(segments):
            n_seg = j - i
            if n_seg < 2 * min_size:
                continue
            seg_mask = segment_indices((i, j))
            seg_bmi = bmi[seg_mask]
            # 候选切分点：动态分位，确保两侧样本数都 >= min_size
            q_lo = min_size / n_seg
            q_hi = 1.0 - min_size / n_seg
            if q_hi <= q_lo:
                continue
            # 步长自适应（至少取5个点）
            n_steps = max(5, int((q_hi - q_lo) / 0.05))
            qs = np.linspace(q_lo, q_hi, n_steps)
            cs = np.unique(np.quantile(seg_bmi, qs))

            for c in cs:
                left = seg_mask & (bmi <= c)
                right = seg_mask & (bmi > c)
                if left.sum() < min_size or right.sum() < min_size:
                    continue
                # 显著性（log-rank）
                try:
                    stat = logrank_test(
                        dur[left],
                        dur[right],
                        event_observed_A=evt[left],
                        event_observed_B=evt[right],
                    )
                    pval = float(stat.p_value)
                except Exception:
                    pval = 1.0
                # 计算新目标值
                new_segs = (
                    segments[:si]
                    + segments[si + 1 :]
                    + [(i, i + left.sum()), (i + left.sum(), j)]
                )
                new_obj = current_objective(new_segs)
                gain = base_obj - new_obj  # 目标下降越多越好（t*更早）

                # 放宽切分准入门槛
                if (pval < 0.10 and gain > 0.02) or (gain > 0.10):
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (si, c, left, right)

        if best_split is None:
            break

        # 应用最佳切分：在原df顺序上，把该段按BMI<=c、>c重排
        si, c, left, right = best_split
        i, j = segments[si]
        # 左右的局部相对顺序不变
        left_idx = np.where(left)[0]
        right_idx = np.where(right)[0]
        new_order = np.concatenate([left_idx, right_idx])
        # 重新排列该段
        for col in ["孕妇BMI", "duration_week", "event"]:
            df.loc[i : j - 1, col] = df.loc[new_order, col].values
            # 同步原数组
        bmi = df["孕妇BMI"].to_numpy(float)
        dur = df["duration_week"].to_numpy(float)
        evt = df["event"].to_numpy(bool)
        # 更新段列表
        left_n = left.sum()
        segments = (
            segments[:si] + [(i, i + left_n), (i + left_n, j)] + segments[si + 1 :]
        )

    # 生成分组编码与标签（按BMI从低到高排序）
    segments = sorted(segments, key=lambda x: df["孕妇BMI"].iloc[x[0] : x[1]].median())
    group_id = np.zeros(len(df), dtype=int)
    labels = []
    boundaries = []
    for k, (i, j) in enumerate(segments, start=1):
        group_id[i:j] = k
        lo = float(df["孕妇BMI"].iloc[i:j].min())
        hi = float(df["孕妇BMI"].iloc[i:j].max())
        labels.append(f"Supervised Q{k}: [{lo:.2f}, {hi:.2f}]")
        boundaries.append(lo)
    boundaries.append(
        float(df["孕妇BMI"].iloc[segments[-1][0] : segments[-1][1]].max())
    )
    df["bmi_group"] = group_id
    df["bmi_group_label"] = pd.Categorical(labels)[group_id - 1].astype(str)

    return df, boundaries


def main():
    # 定位数据与输出目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "Q1", "男胎检测数据_预处理后.csv")
    out_dir = os.path.join(os.path.dirname(__file__), f"output_{GROUPING_METHOD}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading data: {data_path}")
    df_obs_male, df_subj = preprocess_data(data_path, threshold=THRESHOLD)
    print(
        f"Subjects: {df_subj['孕妇代码'].nunique()}, observations: {len(df_obs_male)}"
    )

    # BMI 分组
    if GROUPING_METHOD == "clinical":
        df_subj_g, boundaries = group_bmi_clinical(df_subj)
    elif GROUPING_METHOD == "supervised":
        df_subj_g, boundaries = group_bmi_supervised(
            df_subj, max_groups=MAX_GROUPS, min_size=MIN_GROUP_SIZE, target=SEG_TARGET
        )
    else:
        df_subj_g, boundaries = group_bmi(df_subj, n_groups=5)

    # 保存分组边界
    bnd_df = pd.DataFrame(
        {
            "quantile": list(np.linspace(0, 1, len(boundaries)))[: len(boundaries)],
            "BMI": boundaries + ([] if True else []),
        }
    )
    bnd_path = os.path.join(out_dir, "bmi_boundaries.csv")
    bnd_df.to_csv(bnd_path, index=False, encoding="utf-8-sig")
    print(f"Saved BMI group boundaries: {bnd_path}")

    # 为每个组分析
    results: List[GroupResult] = []
    for g in sorted(df_subj_g["bmi_group"].unique()):
        mask_ids = df_subj_g[df_subj_g["bmi_group"] == g]["孕妇代码"].unique()
        label = df_subj_g[df_subj_g["bmi_group"] == g]["bmi_group_label"].iloc[0]
        df_obs_g = df_obs_male[df_obs_male["孕妇代码"].isin(mask_ids)].copy()
        df_subj_gi = df_subj_g[df_subj_g["bmi_group"] == g].copy()

        res = analyze_group(int(g), str(label), df_obs_g, df_subj_gi, out_dir)
        results.append(res)

    # 汇总表（英文列名）
    res_df = pd.DataFrame(
        [
            {
                "BMI group": r.group_label,
                "Group index": r.group_index,
                "Sample size": r.n,
                "Censoring rate": r.censor_rate,
                "Recommended GA t*": r.t_star,
                "t* 95% CI lower": r.t_star_ci_low,
                "t* 95% CI upper": r.t_star_ci_high,
                "F(t*)": r.F_at_t_star,
                "Window of t*": r.t_star_window,
                "F at 12w": r.F_at_12w,
                "F at 13w": r.F_at_13w,
                "F at 28w": r.F_at_28w,
                "Early alt t (90%)": r.early_alt_t90,
            }
            for r in results
        ]
    ).sort_values("Group index")

    res_path = os.path.join(out_dir, "bmi_groups_results.csv")
    res_df.to_csv(res_path, index=False, encoding="utf-8-sig")
    print(f"Saved results table: {res_path}")

if __name__ == "__main__":
    main()
