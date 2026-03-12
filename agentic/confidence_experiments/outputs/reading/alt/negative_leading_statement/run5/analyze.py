import json
import math
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"


def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / math.sqrt(pooled)


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df.copy()
    df = df[df["speed"].notna() & df["reader_view"].notna() & df["dyslexia_bin"].notna()]

    # Subset to individuals with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Guard against non-positive speeds for log transform
    dys = dys[dys["speed"] > 0]

    rv = dys[dys["reader_view"] == 1]["speed"]
    no_rv = dys[dys["reader_view"] == 0]["speed"]

    # Summary stats
    summary = {
        "n_total": len(dys),
        "n_reader_view": len(rv),
        "n_no_reader_view": len(no_rv),
        "mean_speed_reader_view": float(rv.mean()) if len(rv) else np.nan,
        "mean_speed_no_reader_view": float(no_rv.mean()) if len(no_rv) else np.nan,
        "median_speed_reader_view": float(rv.median()) if len(rv) else np.nan,
        "median_speed_no_reader_view": float(no_rv.median()) if len(no_rv) else np.nan,
    }

    # Tests on log(speed) due to skew
    log_rv = np.log(rv)
    log_no_rv = np.log(no_rv)

    # Welch t-test
    t_res = stats.ttest_ind(log_rv, log_no_rv, equal_var=False, nan_policy="omit")

    # Mann-Whitney U test (two-sided)
    try:
        mw_res = stats.mannwhitneyu(rv, no_rv, alternative="two-sided")
    except ValueError:
        mw_res = None

    # Effect size (Cohen's d on log speed)
    d_log = cohens_d(log_rv, log_no_rv)

    # Regression controlling for covariates
    # Use log(speed) as outcome; include key covariates and page/language fixed effects
    model_df = dys.copy()
    model_df["log_speed"] = np.log(model_df["speed"])

    formula = (
        "log_speed ~ reader_view + num_words + Flesch_Kincaid + "
        "C(page_id) + C(device) + age + C(gender) + C(education) + "
        "C(language) + C(english_native) + correct_rate + img_width + retake_trial"
    )

    try:
        model = smf.ols(formula=formula, data=model_df).fit(cov_type="HC3")
        coef = float(model.params.get("reader_view", np.nan))
        pval = float(model.pvalues.get("reader_view", np.nan))
    except Exception:
        model = None
        coef = np.nan
        pval = np.nan

    results = {
        "summary": summary,
        "t_test_log": {
            "t_stat": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
        },
        "mann_whitney": {
            "u_stat": float(mw_res.statistic) if mw_res is not None else np.nan,
            "p_value": float(mw_res.pvalue) if mw_res is not None else np.nan,
        },
        "effect_size": {
            "cohens_d_log": float(d_log),
        },
        "regression": {
            "coef_reader_view": float(coef),
            "p_value_reader_view": float(pval),
        },
    }

    # Decide response strength
    # Criteria: improvement if reader_view mean > no_reader_view and p<0.05 in regression or t-test.
    mean_diff = summary["mean_speed_reader_view"] - summary["mean_speed_no_reader_view"]

    evidence_yes = False
    if not np.isnan(results["regression"]["p_value_reader_view"]):
        if results["regression"]["p_value_reader_view"] < 0.05 and coef > 0:
            evidence_yes = True
    elif results["t_test_log"]["p_value"] < 0.05 and mean_diff > 0:
        evidence_yes = True

    # Likert mapping
    if evidence_yes:
        # Scale by effect size
        if d_log >= 0.5:
            response = 75
        elif d_log >= 0.2:
            response = 65
        else:
            response = 58
    else:
        # If direction negative or not significant
        if mean_diff <= 0:
            response = 20
        else:
            response = 35

    # Explanation text
    explanation = (
        "Subsetted to participants with dyslexia (dyslexia_bin=1) and compared reading speed between "
        "reader view on vs off. Calculated mean/median speeds, ran Welch t-test on log(speed) to reduce skew, "
        "and a Mann–Whitney test on raw speeds. Also fit an OLS regression on log(speed) with robust (HC3) "
        "standard errors controlling for text difficulty, page, device, age, gender, education, language, "
        "native English, comprehension accuracy, image width, and retake status. "
        f"In this sample (n={summary['n_total']}), mean speed was {summary['mean_speed_reader_view']:.2f} "
        f"with reader view vs {summary['mean_speed_no_reader_view']:.2f} without. "
        f"Welch t-test p={results['t_test_log']['p_value']:.3g}, Mann–Whitney p="
        f"{results['mann_whitney']['p_value']:.3g}, and regression coef="
        f"{results['regression']['coef_reader_view']:.4f} (p={results['regression']['p_value_reader_view']:.3g}). "
        "Given the lack of statistically significant positive effect and/or the direction of the difference, "
        "the evidence does not support that Reader View improves reading speed for individuals with dyslexia."
    )

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    # Also dump results for inspection
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
