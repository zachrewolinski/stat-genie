import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main():
    df = pd.read_csv("reading.csv")
    # Focus on individuals with dyslexia (binary flag)
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Keep only positive speeds for log-transform
    dys = dys[dys["speed"].notna() & (dys["speed"] > 0)].copy()
    dys["log_speed"] = np.log(dys["speed"])

    # Basic counts
    n_rows = len(dys)
    n_participants = dys["uuid"].nunique()
    counts = dys["reader_view"].value_counts().to_dict()

    # How many participants saw both conditions?
    both = dys.groupby("uuid")["reader_view"].nunique()
    n_both = int((both == 2).sum())

    # Unadjusted difference in log-speed
    log_means = dys.groupby("reader_view")["log_speed"].mean()
    log_means = log_means.to_dict()
    # Welch t-test (ignores clustering; reported as descriptive)
    rv1 = dys[dys["reader_view"] == 1]["log_speed"]
    rv0 = dys[dys["reader_view"] == 0]["log_speed"]
    t_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

    # Cluster-robust OLS with controls
    # Keep a reasonable set of covariates for reading speed
    covariates = [
        "reader_view",
        "page_id",
        "num_words",
        "device",
        "age",
        "gender",
        "education",
        "english_native",
        "retake_trial",
        "correct_rate",
        "Flesch_Kincaid",
        "uuid",
    ]
    model_df = dys.dropna(subset=covariates + ["log_speed"]).copy()

    formula = (
        "log_speed ~ reader_view + C(page_id) + num_words + C(device) + age + "
        "C(gender) + C(education) + C(english_native) + retake_trial + correct_rate + Flesch_Kincaid"
    )

    model = smf.ols(formula, data=model_df).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["uuid"]}
    )

    coef = model.params.get("reader_view", np.nan)
    pval = model.pvalues.get("reader_view", np.nan)
    # Convert log-effect to percent change
    pct_change = (np.exp(coef) - 1.0) * 100.0 if np.isfinite(coef) else np.nan

    # Also compute raw speed summaries for interpretability
    speed_summary = dys.groupby("reader_view")["speed"].agg(["mean", "median", "count"]).to_dict()

    results = {
        "n_rows": n_rows,
        "n_participants": n_participants,
        "reader_view_counts": counts,
        "n_participants_both_conditions": n_both,
        "log_speed_means": log_means,
        "welch_t_pvalue_log": float(t_res.pvalue),
        "model_n": int(model.nobs),
        "coef_reader_view_log": float(coef),
        "pvalue_reader_view_log": float(pval),
        "pct_change_reader_view": float(pct_change),
        "speed_summary": speed_summary,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
