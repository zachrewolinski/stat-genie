import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "reading.csv"


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def main():
    df = pd.read_csv(DATA_PATH)
    # Focus on participants with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Basic cleaning: remove non-positive or missing speed values
    dys = dys.replace([np.inf, -np.inf], np.nan)
    dys = dys.dropna(subset=["speed", "reader_view"])
    dys = dys[dys["speed"] > 0]

    # Log-transform speed to reduce skew
    dys["log_speed"] = np.log(dys["speed"])

    # Remove extreme outliers on log scale (1st-99th percentile)
    lo, hi = dys["log_speed"].quantile([0.01, 0.99])
    dys_trim = dys[(dys["log_speed"] >= lo) & (dys["log_speed"] <= hi)].copy()

    rv1 = dys_trim[dys_trim["reader_view"] == 1]["log_speed"]
    rv0 = dys_trim[dys_trim["reader_view"] == 0]["log_speed"]

    # Welch t-test on log speed
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False)

    # Mann-Whitney U (nonparametric) on log speed
    try:
        u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan

    d_log = cohen_d(rv1, rv0)

    # Regression with cluster-robust SE by participant uuid
    # Use available covariates to control for text and participant/device characteristics
    covariates = [
        "reader_view",
        "num_words",
        "Flesch_Kincaid",
        "device",
        "language",
        "age",
        "gender",
        "education",
        "english_native",
        "retake_trial",
        "page_id",
    ]

    reg_df = dys_trim.dropna(subset=covariates + ["log_speed", "uuid"]).copy()

    # One-hot encode categoricals
    reg_df = pd.get_dummies(
        reg_df,
        columns=["device", "language", "education", "english_native", "page_id"],
        drop_first=True,
    )

    y = reg_df["log_speed"]
    X = reg_df.drop(columns=["log_speed", "uuid"])
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": reg_df["uuid"]})

    coef = results.params.get("reader_view", np.nan)
    se = results.bse.get("reader_view", np.nan)
    pval = results.pvalues.get("reader_view", np.nan)

    # Convert log-coefficient to percent change for interpretability
    pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

    output = {
        "n_dyslexia_rows": int(len(dys)),
        "n_dyslexia_trim": int(len(dys_trim)),
        "mean_log_speed_reader_view": float(rv1.mean()) if len(rv1) else np.nan,
        "mean_log_speed_no_reader_view": float(rv0.mean()) if len(rv0) else np.nan,
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "u_stat": float(u_stat),
        "u_p": float(u_p),
        "cohen_d_log": float(d_log),
        "reg_coef_reader_view": float(coef),
        "reg_se_reader_view": float(se),
        "reg_p_reader_view": float(pval),
        "reg_pct_change_reader_view": float(pct_change),
        "reg_n": int(len(reg_df)),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
