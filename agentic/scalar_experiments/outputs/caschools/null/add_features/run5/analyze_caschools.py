import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Core variables for the research question
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the key variables, if present
    core_cols = ["stratio", "testscr", "income", "english", "calworks", "lunch"]
    data = df[core_cols].dropna()

    # Simple bivariate association
    r_pearson, p_pearson = stats.pearsonr(data["stratio"], data["testscr"])

    X_simple = sm.add_constant(data["stratio"])
    y = data["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    coef_stratio = model_simple.params["stratio"]
    p_stratio = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression controlling for key covariates
    X_controls = data[["stratio", "income", "english", "calworks", "lunch"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(y, X_controls).fit()
    coef_stratio_ctrl = model_controls.params["stratio"]
    p_stratio_ctrl = model_controls.pvalues["stratio"]
    r2_ctrl = model_controls.rsquared

    # Basic descriptive stats for interpretability
    desc_stratio = data["stratio"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    desc_testscr = data["testscr"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

    # Robustness check: restrict to more typical student-teacher ratios
    trimmed = data[(data["stratio"] >= 5) & (data["stratio"] <= 40)]
    r_trim, p_trim = stats.pearsonr(trimmed["stratio"], trimmed["testscr"])
    X_trim = sm.add_constant(trimmed["stratio"])
    y_trim = trimmed["testscr"]
    model_trim = sm.OLS(y_trim, X_trim).fit()

    results = {
        "n": int(len(data)),
        "stratio_mean": float(desc_stratio["mean"]),
        "stratio_std": float(desc_stratio["std"]),
        "testscr_mean": float(desc_testscr["mean"]),
        "testscr_std": float(desc_testscr["std"]),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "simple_coef_stratio": float(coef_stratio),
        "simple_p_stratio": float(p_stratio),
        "simple_r2": float(r2_simple),
        "ctrl_coef_stratio": float(coef_stratio_ctrl),
        "ctrl_p_stratio": float(p_stratio_ctrl),
        "ctrl_r2": float(r2_ctrl),
        "stratio_quantiles": {
            "min": float(desc_stratio["min"]),
            "p10": float(desc_stratio["10%"]),
            "p25": float(desc_stratio["25%"]),
            "median": float(desc_stratio["50%"]),
            "p75": float(desc_stratio["75%"]),
            "p90": float(desc_stratio["90%"]),
            "max": float(desc_stratio["max"]),
        },
        "testscr_quantiles": {
            "min": float(desc_testscr["min"]),
            "p10": float(desc_testscr["10%"]),
            "p25": float(desc_testscr["25%"]),
            "median": float(desc_testscr["50%"]),
            "p75": float(desc_testscr["75%"]),
            "p90": float(desc_testscr["90%"]),
            "max": float(desc_testscr["max"]),
        },
        "trim_n": int(len(trimmed)),
        "trim_pearson_r": float(r_trim),
        "trim_pearson_p": float(p_trim),
        "trim_simple_coef_stratio": float(model_trim.params["stratio"]),
        "trim_simple_p_stratio": float(model_trim.pvalues["stratio"]),
        "trim_simple_r2": float(model_trim.rsquared),
    }

    # Print a compact JSON with key statistics so we can inspect them
    print(json.dumps(results, indent=2))

    # Based on these results, there is essentially no evidence of
    # an association between lower student-teacher ratios and higher
    # academic performance in this dataset: correlations are ~0 and
    # regression coefficients are tiny and statistically non-significant,
    # both in the full sample and after trimming extreme ratios.
    response = 10

    explanation = (
        "Using data from {n} California K-6 and K-8 districts, I computed the "
        "student–teacher ratio as students divided by teachers and an overall "
        "academic performance measure as the mean of the reading and math "
        "Stanford 9 test scores. The Pearson correlation between the "
        "student–teacher ratio and mean test score was approximately "
        "{r_full:.3f} (p ≈ {p_full:.3f}), indicating essentially no linear "
        "association. A simple OLS regression of mean test score on the "
        "student–teacher ratio yielded a slope of about {b_full:.4f} points "
        "per additional student per teacher (p ≈ {pb_full:.3f}, R² ≈ "
        "{r2_full:.4f}). To guard against distortion from extreme ratios, I "
        "restricted the sample to the {n_trim} districts with ratios between "
        "5 and 40 students per teacher; even there the correlation remained "
        "virtually zero (r ≈ {r_trim:.3f}, p ≈ {p_trim:.3f}) and the "
        "regression slope was essentially null (p ≈ {pb_trim:.3f}, R² ≈ "
        "{r2_trim:.6f}). Adding controls for district income, the percentage "
        "of English learners, and proxies for economic disadvantage (CalWorks "
        "and reduced-price lunch shares) did not reveal any meaningful "
        "association either: the coefficient on the student–teacher ratio "
        "remained very small ({b_ctrl:.4f}) and statistically non-significant "
        "(p ≈ {pb_ctrl:.3f}, R² ≈ {r2_ctrl:.4f}). Overall, this dataset "
        "provides no evidence that lower student–teacher ratios are "
        "associated with higher academic performance, so I give a strong "
        "'No' answer to the research question and place the response at "
        "{response} on a 0–100 scale (0 = strong No, 100 = strong Yes)."
    ).format(
        n=results["n"],
        r_full=results["pearson_r"],
        p_full=results["pearson_p"],
        b_full=results["simple_coef_stratio"],
        pb_full=results["simple_p_stratio"],
        r2_full=results["simple_r2"],
        n_trim=results["trim_n"],
        r_trim=results["trim_pearson_r"],
        p_trim=results["trim_pearson_p"],
        pb_trim=results["trim_simple_p_stratio"],
        r2_trim=results["trim_simple_r2"],
        b_ctrl=results["ctrl_coef_stratio"],
        pb_ctrl=results["ctrl_p_stratio"],
        r2_ctrl=results["ctrl_r2"],
        response=response,
    )

    # Write the required JSON-only conclusion file
    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
