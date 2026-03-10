import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("reading.csv")

    # Focus on participants with dyslexia
    if "dyslexia_bin" in df.columns:
        dys = df[df["dyslexia_bin"] == 1].copy()
    else:
        dys = df[df["dyslexia"] > 0].copy()

    # Basic sanity
    dys = dys.dropna(subset=["speed", "reader_view"])

    # Group means
    grp = dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

    # Welch t-test on raw speed
    rv0 = dys.loc[dys["reader_view"] == 0, "speed"].values
    rv1 = dys.loc[dys["reader_view"] == 1, "speed"].values
    t_raw = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

    # Log-speed to reduce skew
    dys = dys[dys["speed"] > 0].copy()
    dys["log_speed"] = np.log(dys["speed"])
    rv0_log = dys.loc[dys["reader_view"] == 0, "log_speed"].values
    rv1_log = dys.loc[dys["reader_view"] == 1, "log_speed"].values
    t_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy="omit")

    # Regression with controls and cluster-robust SEs by participant
    # Keep rows with required columns
    model_cols = [
        "log_speed",
        "reader_view",
        "page_id",
        "num_words",
        "language",
        "device",
        "age",
        "gender",
        "education",
        "english_native",
        "retake_trial",
        "uuid",
    ]
    present_cols = [c for c in model_cols if c in dys.columns]
    reg_df = dys[present_cols].dropna().copy()

    # If too many rows drop due to missing, fallback to smaller model
    if len(reg_df) < max(50, 0.5 * len(dys)):
        fallback_cols = [
            "log_speed",
            "reader_view",
            "page_id",
            "num_words",
            "uuid",
        ]
        present_cols = [c for c in fallback_cols if c in dys.columns]
        reg_df = dys[present_cols].dropna().copy()

    # Build formula
    # Categorical controls as C() in formula
    formula_parts = ["reader_view"]
    for c in ["page_id", "language", "device", "education", "english_native"]:
        if c in reg_df.columns:
            formula_parts.append(f"C({c})")
    for c in ["num_words", "age", "gender", "retake_trial"]:
        if c in reg_df.columns:
            formula_parts.append(c)
    formula = "log_speed ~ " + " + ".join(formula_parts)

    # Fit with clustered SEs if uuid available
    if "uuid" in reg_df.columns:
        model = smf.ols(formula, data=reg_df).fit(cov_type="cluster", cov_kwds={"groups": reg_df["uuid"]})
    else:
        model = smf.ols(formula, data=reg_df).fit()

    coef = model.params.get("reader_view", np.nan)
    pval = model.pvalues.get("reader_view", np.nan)
    conf = model.conf_int().loc["reader_view"].tolist() if "reader_view" in model.params.index else [np.nan, np.nan]

    # Compute percent change from log coefficient
    pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

    results = {
        "n_dyslexia_rows": int(len(dys)),
        "group_stats": grp.to_dict(orient="records"),
        "t_raw": {"stat": float(t_raw.statistic), "p": float(t_raw.pvalue)},
        "t_log": {"stat": float(t_log.statistic), "p": float(t_log.pvalue)},
        "regression": {
            "n": int(len(reg_df)),
            "coef_reader_view_log": float(coef),
            "p_reader_view": float(pval),
            "conf_low": float(conf[0]),
            "conf_high": float(conf[1]),
            "pct_change_speed": float(pct_change),
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
