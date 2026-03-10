import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

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
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv(DATA_PATH)

    # Ensure numeric
    df = df.copy()

    # Define dyslexia subset
    if "dyslexia_bin" in df.columns:
        dys = df[df["dyslexia_bin"] == 1].copy()
    else:
        dys = df[df["dyslexia"] >= 1].copy()

    # Basic cleaning
    dys = dys.replace([np.inf, -np.inf], np.nan)
    dys = dys.dropna(subset=["speed", "reader_view", "uuid"])
    dys = dys[dys["speed"] > 0]

    dys["log_speed"] = np.log(dys["speed"])

    # Group stats
    g0 = dys[dys["reader_view"] == 0]["log_speed"]
    g1 = dys[dys["reader_view"] == 1]["log_speed"]

    # Welch t-test on log speed
    t_stat, t_p = stats.ttest_ind(g1, g0, equal_var=False, nan_policy="omit")

    # Mann-Whitney U on raw speed (two-sided)
    try:
        u_stat, u_p = stats.mannwhitneyu(
            dys[dys["reader_view"] == 1]["speed"],
            dys[dys["reader_view"] == 0]["speed"],
            alternative="two-sided",
        )
    except Exception:
        u_stat, u_p = np.nan, np.nan

    d = cohen_d(g1, g0)

    # Paired analysis if participants saw both conditions
    paired = dys.groupby(["uuid", "reader_view"])['log_speed'].mean().reset_index()
    pivot = paired.pivot(index="uuid", columns="reader_view", values="log_speed")
    pivot = pivot.dropna()
    paired_n = pivot.shape[0]

    if paired_n >= 2:
        diff = pivot[1] - pivot[0]
        t_stat_p, t_p_p = stats.ttest_rel(pivot[1], pivot[0], nan_policy="omit")
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except Exception:
            w_stat, w_p = np.nan, np.nan
    else:
        diff = pd.Series(dtype=float)
        t_stat_p, t_p_p, w_stat, w_p = np.nan, np.nan, np.nan, np.nan

    # Regression with cluster-robust SE by uuid
    # Use log_speed ~ reader_view + C(page_id) + num_words (if available)
    reg_df = dys.dropna(subset=["log_speed", "reader_view", "page_id", "num_words", "uuid"]).copy()
    reg_result = None
    if reg_df.shape[0] >= 10:
        try:
            model = smf.ols("log_speed ~ reader_view + C(page_id) + num_words", data=reg_df)
            reg_result = model.fit(cov_type="cluster", cov_kwds={"groups": reg_df["uuid"]})
        except Exception:
            reg_result = None

    summary = {
        "n_dyslexic_rows": int(dys.shape[0]),
        "n_dyslexic_unique": int(dys["uuid"].nunique()),
        "reader_view_counts": dys["reader_view"].value_counts().to_dict(),
        "log_speed_mean_reader_view_1": float(g1.mean()) if len(g1) else np.nan,
        "log_speed_mean_reader_view_0": float(g0.mean()) if len(g0) else np.nan,
        "speed_median_reader_view_1": float(dys[dys["reader_view"] == 1]["speed"].median()) if len(g1) else np.nan,
        "speed_median_reader_view_0": float(dys[dys["reader_view"] == 0]["speed"].median()) if len(g0) else np.nan,
        "welch_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
        "welch_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "mannwhitney_p": float(u_p) if np.isfinite(u_p) else np.nan,
        "cohen_d": float(d) if np.isfinite(d) else np.nan,
        "paired_n": int(paired_n),
        "paired_t_p": float(t_p_p) if np.isfinite(t_p_p) else np.nan,
        "paired_t_stat": float(t_stat_p) if np.isfinite(t_stat_p) else np.nan,
        "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
    }

    if reg_result is not None:
        coef = reg_result.params.get("reader_view", np.nan)
        pval = reg_result.pvalues.get("reader_view", np.nan)
        summary.update({
            "reg_reader_view_coef": float(coef) if np.isfinite(coef) else np.nan,
            "reg_reader_view_p": float(pval) if np.isfinite(pval) else np.nan,
            "reg_n": int(reg_df.shape[0])
        })

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
