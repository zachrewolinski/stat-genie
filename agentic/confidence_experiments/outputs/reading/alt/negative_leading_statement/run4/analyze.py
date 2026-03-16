import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

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

    # Focus on individuals with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Basic cleaning
    dys = dys.replace([np.inf, -np.inf], np.nan)
    dys = dys.dropna(subset=["speed", "reader_view", "uuid"])

    # Ensure numeric
    dys["speed"] = pd.to_numeric(dys["speed"], errors="coerce")
    dys = dys.dropna(subset=["speed"])

    # Log speed for modeling
    dys["log_speed"] = np.log(dys["speed"])

    # Group summaries
    grp = dys.groupby("reader_view")
    summary = grp["speed"].agg(["count", "mean", "median", "std"]).to_dict()

    # T-tests (Welch) for raw and log speed
    speed0 = dys.loc[dys["reader_view"] == 0, "speed"].values
    speed1 = dys.loc[dys["reader_view"] == 1, "speed"].values
    log0 = dys.loc[dys["reader_view"] == 0, "log_speed"].values
    log1 = dys.loc[dys["reader_view"] == 1, "log_speed"].values

    t_raw = stats.ttest_ind(speed1, speed0, equal_var=False, nan_policy="omit")
    t_log = stats.ttest_ind(log1, log0, equal_var=False, nan_policy="omit")
    mw = stats.mannwhitneyu(speed1, speed0, alternative="two-sided")

    d_raw = cohen_d(speed1, speed0)
    d_log = cohen_d(log1, log0)

    # Regression models
    # Simple model
    m1 = smf.ols("log_speed ~ reader_view", data=dys).fit(
        cov_type="cluster", cov_kwds={"groups": dys["uuid"]}
    )

    # Controlled model with key covariates
    formula = (
        "log_speed ~ reader_view + num_words + C(page_id) + C(device) + age + C(gender) + "
        "C(education) + C(language) + C(english_native) + retake_trial + Flesch_Kincaid + correct_rate"
    )
    needed_cols = [
        "log_speed",
        "reader_view",
        "num_words",
        "page_id",
        "device",
        "age",
        "gender",
        "education",
        "language",
        "english_native",
        "retake_trial",
        "Flesch_Kincaid",
        "correct_rate",
        "uuid",
    ]
    dys_m2 = dys[needed_cols].dropna()
    m2 = smf.ols(formula, data=dys_m2).fit(
        cov_type="cluster", cov_kwds={"groups": dys_m2["uuid"]}
    )

    def pct_effect(beta):
        return (np.exp(beta) - 1) * 100

    results = {
        "n_total": int(len(dys)),
        "n_m2": int(len(dys_m2)),
        "summary": summary,
        "t_raw": {"stat": float(t_raw.statistic), "p": float(t_raw.pvalue)},
        "t_log": {"stat": float(t_log.statistic), "p": float(t_log.pvalue)},
        "mw": {"stat": float(mw.statistic), "p": float(mw.pvalue)},
        "cohen_d": {"raw": float(d_raw), "log": float(d_log)},
        "m1": {
            "beta_reader_view": float(m1.params.get("reader_view", np.nan)),
            "p_reader_view": float(m1.pvalues.get("reader_view", np.nan)),
            "pct_effect": float(pct_effect(m1.params.get("reader_view", np.nan))),
        },
        "m2": {
            "beta_reader_view": float(m2.params.get("reader_view", np.nan)),
            "p_reader_view": float(m2.pvalues.get("reader_view", np.nan)),
            "pct_effect": float(pct_effect(m2.params.get("reader_view", np.nan))),
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
