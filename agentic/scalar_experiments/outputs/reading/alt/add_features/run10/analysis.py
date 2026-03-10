import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"


def cohen_d(x, y):
    # Cohen's d for independent samples (pooled SD)
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

    # Determine dyslexia indicator
    if "dyslexia_bin" in df.columns:
        dyslexia_mask = df["dyslexia_bin"] == 1
        dyslexia_def = "dyslexia_bin == 1"
    else:
        dyslexia_mask = df["dyslexia"] > 0
        dyslexia_def = "dyslexia > 0"

    # Core variables
    use_cols = [
        "speed",
        "reader_view",
        "num_words",
        "page_id",
        "language",
        "device",
        "age",
        "gender",
        "education",
        "english_native",
        "correct_rate",
        "retake_trial",
    ]
    use_cols = [c for c in use_cols if c in df.columns]

    dfd = df.loc[dyslexia_mask, use_cols].copy()

    # Clean
    dfd = dfd.dropna(subset=["speed", "reader_view"]).copy()
    dfd["reader_view"] = dfd["reader_view"].astype(int)

    # Basic descriptives
    desc = dfd.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"])

    # Welch t-test on log(speed)
    dfd = dfd[dfd["speed"] > 0].copy()
    dfd["log_speed"] = np.log(dfd["speed"])

    group0 = dfd[dfd["reader_view"] == 0]["log_speed"].values
    group1 = dfd[dfd["reader_view"] == 1]["log_speed"].values

    t_res = stats.ttest_ind(group1, group0, equal_var=False, nan_policy="omit")
    d_log = cohen_d(group1, group0)

    # Regression with covariates, robust SE
    # Build formula dynamically depending on available columns
    cov_terms = []
    for term in [
        "num_words",
        "age",
        "correct_rate",
        "retake_trial",
    ]:
        if term in dfd.columns:
            cov_terms.append(term)

    for term in ["page_id", "language", "device", "gender", "education", "english_native"]:
        if term in dfd.columns:
            cov_terms.append(f"C({term})")

    if cov_terms:
        formula = "log_speed ~ reader_view + " + " + ".join(cov_terms)
    else:
        formula = "log_speed ~ reader_view"

    model_df = dfd.dropna(subset=["log_speed", "reader_view"]).copy()
    # Drop rows with missing covariates used in formula
    model_df = model_df.dropna()

    model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    coef = model.params.get("reader_view", np.nan)
    pval = model.pvalues.get("reader_view", np.nan)

    results = {
        "dyslexia_definition": dyslexia_def,
        "n_dyslexia": int(dfd.shape[0]),
        "desc_speed_by_reader_view": desc.reset_index().to_dict(orient="records"),
        "t_test_log_speed": {
            "t_stat": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
            "cohen_d_log": float(d_log),
            "n_reader_view_1": int(len(group1)),
            "n_reader_view_0": int(len(group0)),
        },
        "regression": {
            "formula": formula,
            "n": int(model_df.shape[0]),
            "coef_reader_view": float(coef),
            "p_value_reader_view": float(pval),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
