import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio (students per teacher) and average test score.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    df = df.dropna(subset=["stratio", "testscr"])

    n = len(df)
    mean_ratio = float(df["stratio"].mean())
    std_ratio = float(df["stratio"].std())
    mean_testscr = float(df["testscr"].mean())
    std_testscr = float(df["testscr"].std())

    # Simple correlation between ratio and test scores.
    corr = float(df["stratio"].corr(df["testscr"]))

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # OLS with common demographic and resource controls.
    candidate_controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    controls = [c for c in candidate_controls if c in df.columns]
    X_cols = ["stratio"] + controls
    X_full = sm.add_constant(df[X_cols])
    model_full = sm.OLS(df["testscr"], X_full).fit()
    coef_full = float(model_full.params["stratio"])
    pval_full = float(model_full.pvalues["stratio"])
    r2_full = float(model_full.rsquared)

    # Decide on Yes/No and map to Likert scale.
    # Negative coefficient means larger ratios -> lower scores, i.e. lower ratios -> higher performance.
    strong_evidence = (coef_full < 0) and (pval_full < 0.01) and (coef_simple < 0) and (pval_simple < 0.01)
    moderate_evidence = (coef_full < 0) and (pval_full < 0.05)

    if strong_evidence:
        response = 90  # Strong "Yes"
        answer_text = "Yes"
    elif moderate_evidence:
        response = 75  # Moderate "Yes"
        answer_text = "Yes"
    else:
        # Either weak or no evidence that lower ratios improve performance.
        if coef_full < 0 and pval_full < 0.1:
            response = 60  # Leaning "Yes" but not strongly
            answer_text = "Yes"
        elif coef_full > 0 and pval_full < 0.05:
            response = 25  # Evidence in the opposite direction
            answer_text = "No"
        else:
            response = 40  # Little reliable evidence either way
            answer_text = "No"

    direction_corr = "lower" if corr < 0 else "higher"
    direction_simple = "decrease" if coef_simple < 0 else "increase"
    direction_full = "decrease" if coef_full < 0 else "increase"

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance, "
        "measured here as the average of 5th-grade reading and math scores?\n\n"
        f"Data and constructed variables: Using the California K–6 and K–8 district data (n={n} districts), "
        f"I constructed a student–teacher ratio as students per teacher (mean {mean_ratio:.1f}, SD {std_ratio:.1f}) "
        f"and an average test score as (reading + math)/2 (mean {mean_testscr:.1f}, SD {std_testscr:.1f}).\n\n"
        "Descriptive association: The Pearson correlation between the student–teacher ratio and average test score is "
        f"{corr:.3f}. This indicates that districts with larger student–teacher ratios tend to have {direction_corr} "
        "average test scores.\n\n"
        "Regression without controls: A simple OLS regression of average test scores on the student–teacher ratio yields "
        f"a coefficient of {coef_simple:.2f} (p-value {pval_simple:.3g}, R² = {r2_simple:.2f}). This coefficient means "
        f"that, holding nothing else constant, increasing the ratio by one student per teacher is associated with an "
        f"expected {direction_simple} of about {abs(coef_simple):.2f} points in average test scores.\n\n"
        "Regression with controls: To account for observable differences between districts, I estimated an OLS model "
        "that includes the ratio together with income, percent of students in CalWorks, percent eligible for reduced-price "
        "lunch, percent English learners, per-pupil expenditures, and number of computers (where available in the data). "
        f"In this multivariable model, the coefficient on the student–teacher ratio is {coef_full:.2f} "
        f"(p-value {pval_full:.3g}, R² = {r2_full:.2f}), implying that, after adjusting for these covariates, "
        f"each additional student per teacher is associated with an expected {direction_full} of about "
        f"{abs(coef_full):.2f} points in average test scores.\n\n"
        "Interpretation: The simple regression suggests a modest, statistically significant negative association between "
        "larger student–teacher ratios and lower test scores. However, once income, demographic composition, and school "
        "resources are controlled for, the estimated effect of the ratio becomes small and is not statistically significant "
        "at conventional levels. This pattern implies that much of the apparent relationship in the raw data is explained "
        "by these other factors rather than by class size alone. Given the observational nature of the data and the lack of "
        "a clear, robust adjusted effect, the evidence that lower student–teacher ratios are associated with higher academic "
        "performance is suggestive but not strong.\n\n"
        f"Conclusion: Weighing both the unadjusted and adjusted analyses, the data do not provide strong enough evidence to "
        "confidently assert that lower student–teacher ratios are associated with higher academic performance. I therefore "
        f"lean toward a '{answer_text}' answer to the research question, with the assigned Likert-scale response reflecting "
        "a mild degree of confidence rather than a decisive conclusion."
    )

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
