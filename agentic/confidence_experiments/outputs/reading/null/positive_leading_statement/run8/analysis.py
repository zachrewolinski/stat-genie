import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


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
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv("reading.csv")

    # Focus on participants with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Basic cleaning: keep finite speeds
    dys = dys[np.isfinite(dys["speed"])].copy()

    # Split by reader view
    rv1 = dys[dys["reader_view"] == 1]
    rv0 = dys[dys["reader_view"] == 0]

    # Summary stats
    summary = {
        "n_total": int(dys.shape[0]),
        "n_reader_view_on": int(rv1.shape[0]),
        "n_reader_view_off": int(rv0.shape[0]),
        "mean_speed_on": float(rv1["speed"].mean()),
        "mean_speed_off": float(rv0["speed"].mean()),
        "median_speed_on": float(rv1["speed"].median()),
        "median_speed_off": float(rv0["speed"].median()),
    }

    # Welch t-test on raw speed
    t_res = stats.ttest_ind(rv1["speed"], rv0["speed"], equal_var=False, nan_policy="omit")

    # Mann-Whitney U (two-sided)
    try:
        mw_res = stats.mannwhitneyu(rv1["speed"], rv0["speed"], alternative="two-sided")
        mw_u = float(mw_res.statistic)
        mw_p = float(mw_res.pvalue)
    except Exception:
        mw_u = np.nan
        mw_p = np.nan

    # Effect size
    d = cohen_d(rv1["speed"], rv0["speed"])

    # Log-speed regression with controls
    # Avoid log(0) by filtering positive speeds
    dys_pos = dys[dys["speed"] > 0].copy()
    dys_pos["log_speed"] = np.log(dys_pos["speed"])

    # Encode categorical controls
    # Use page_id, device, language as controls (fixed effects)
    model = smf.ols(
        "log_speed ~ reader_view + C(page_id) + C(device) + C(language)",
        data=dys_pos,
    ).fit(cov_type="HC3")

    results = {
        "summary": summary,
        "t_stat": float(t_res.statistic),
        "t_p": float(t_res.pvalue),
        "mw_u": mw_u,
        "mw_p": mw_p,
        "cohen_d": float(d) if np.isfinite(d) else None,
        "reg_coef_reader_view": float(model.params.get("reader_view", np.nan)),
        "reg_p_reader_view": float(model.pvalues.get("reader_view", np.nan)),
        "reg_n": int(model.nobs),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
