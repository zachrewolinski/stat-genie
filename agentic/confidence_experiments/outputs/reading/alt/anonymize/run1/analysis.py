import json
import math
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (x.mean() - y.mean()) / math.sqrt(pooled)


def main():
    df = pd.read_csv("reading.csv")

    # Define reading speed (words per minute) using time-on-page minus scrolling time
    df = df.rename(columns={
        "feature3": "reader_view",
        "feature5": "reading_time_ms",
        "feature7": "word_count",
        "feature16": "retake",
        "feature17": "dyslexia",
    })

    df = df.copy()
    df = df[df["reading_time_ms"].notna() & df["word_count"].notna()]
    df = df[df["reading_time_ms"] > 0]
    df["speed_wpm"] = df["word_count"] / (df["reading_time_ms"] / 60000.0)

    # Remove extreme outliers that are likely timing artifacts
    lower, upper = df["speed_wpm"].quantile([0.01, 0.99])
    df = df[(df["speed_wpm"] >= lower) & (df["speed_wpm"] <= upper)]

    # Focus on participants with dyslexia; exclude retakes to avoid learning effects
    df = df[(df["dyslexia"] == 1) & (df["retake"] == 0)]

    group_on = df[df["reader_view"] == 1]["speed_wpm"]
    group_off = df[df["reader_view"] == 0]["speed_wpm"]

    summary = {
        "n_on": int(group_on.shape[0]),
        "n_off": int(group_off.shape[0]),
        "mean_on": float(group_on.mean()),
        "mean_off": float(group_off.mean()),
        "median_on": float(group_on.median()),
        "median_off": float(group_off.median()),
        "std_on": float(group_on.std(ddof=1)),
        "std_off": float(group_off.std(ddof=1)),
    }

    # Welch t-test
    t_res = stats.ttest_ind(group_on, group_off, equal_var=False, nan_policy="omit")

    # Mann-Whitney U test
    try:
        u_res = stats.mannwhitneyu(group_on, group_off, alternative="two-sided")
        u_stat = float(u_res.statistic)
        u_p = float(u_res.pvalue)
    except Exception:
        u_stat = float("nan")
        u_p = float("nan")

    d = cohen_d(group_on, group_off)

    # Simple regression: speed ~ reader_view
    X = sm.add_constant(df["reader_view"].astype(float))
    y = df["speed_wpm"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="HC3")

    results = {
        "summary": summary,
        "t_stat": float(t_res.statistic),
        "t_p": float(t_res.pvalue),
        "u_stat": u_stat,
        "u_p": u_p,
        "cohen_d": float(d),
        "reg_coef_reader_view": float(model.params.get("reader_view", np.nan)),
        "reg_p_reader_view": float(model.pvalues.get("reader_view", np.nan)),
        "reg_n": int(model.nobs),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
