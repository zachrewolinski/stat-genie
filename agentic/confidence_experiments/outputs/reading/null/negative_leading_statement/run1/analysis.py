import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def summarize_group(df, label):
    return {
        "label": label,
        "n": int(df.shape[0]),
        "mean": float(df["speed"].mean()),
        "median": float(df["speed"].median()),
        "std": float(df["speed"].std(ddof=1)),
    }


def main():
    df = pd.read_csv("reading.csv")

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["speed", "reader_view", "dyslexia_bin"])  # required
    df = df[df["speed"] > 0]

    # Define dyslexic subset
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Groups by reader view
    g0 = dys[dys["reader_view"] == 0]["speed"]
    g1 = dys[dys["reader_view"] == 1]["speed"]

    # Summary stats
    summaries = {
        "reader_view_0": summarize_group(dys[dys["reader_view"] == 0], "reader_view=0"),
        "reader_view_1": summarize_group(dys[dys["reader_view"] == 1], "reader_view=1"),
    }

    # Welch t-test on raw speed
    t_raw = stats.ttest_ind(g1, g0, equal_var=False, nan_policy="omit")

    # Mann-Whitney U
    u_raw = stats.mannwhitneyu(g1, g0, alternative="two-sided")

    # Log-speed analysis
    dys = dys.copy()
    dys["log_speed"] = np.log(dys["speed"])
    lg0 = dys[dys["reader_view"] == 0]["log_speed"]
    lg1 = dys[dys["reader_view"] == 1]["log_speed"]
    t_log = stats.ttest_ind(lg1, lg0, equal_var=False, nan_policy="omit")

    d_raw = cohens_d(g1, g0)
    d_log = cohens_d(lg1, lg0)

    # Regression on log_speed with controls and cluster-robust SE by uuid
    # Keep rows with required covariates
    covars = ["num_words", "page_id", "device", "age", "correct_rate", "retake_trial", "english_native"]
    reg_df = dys.dropna(subset=["log_speed", "reader_view"] + covars + ["uuid"])

    # Limit categories that might be sparse by keeping as is; statsmodels handles with C()
    formula = "log_speed ~ reader_view + num_words + C(page_id) + C(device) + age + correct_rate + retake_trial + C(english_native)"
    model = smf.ols(formula, data=reg_df).fit(cov_type="cluster", cov_kwds={"groups": reg_df["uuid"]})
    coef = model.params.get("reader_view", np.nan)
    pval = model.pvalues.get("reader_view", np.nan)

    # Convert log-coef to percent change
    pct_change = (np.exp(coef) - 1.0) if np.isfinite(coef) else np.nan

    results = {
        "n_total": int(df.shape[0]),
        "n_dyslexic": int(dys.shape[0]),
        "summaries": summaries,
        "t_raw": {"stat": float(t_raw.statistic), "pvalue": float(t_raw.pvalue)},
        "u_raw": {"stat": float(u_raw.statistic), "pvalue": float(u_raw.pvalue)},
        "t_log": {"stat": float(t_log.statistic), "pvalue": float(t_log.pvalue)},
        "cohens_d_raw": float(d_raw) if np.isfinite(d_raw) else None,
        "cohens_d_log": float(d_log) if np.isfinite(d_log) else None,
        "regression": {
            "n": int(reg_df.shape[0]),
            "coef_reader_view_log": float(coef),
            "pvalue_reader_view": float(pval),
            "pct_change_reader_view": float(pct_change),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
