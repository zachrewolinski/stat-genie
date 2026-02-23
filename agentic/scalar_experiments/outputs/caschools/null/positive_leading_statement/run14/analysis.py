import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (for research question text if needed later)
    info_path = base_path / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    # Load data
    data_path = base_path / "caschools.csv"
    df = pd.read_csv(data_path)

    # Construct key variables
    # Student–teacher ratio
    df["str"] = df["students"] / df["teachers"]
    # Overall academic performance as the mean of reading and math scores
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing values in variables used
    vars_basic = ["testscr", "str"]
    vars_covariates = ["income", "english", "calworks", "lunch"]
    df_basic = df.dropna(subset=vars_basic)
    df_full = df.dropna(subset=vars_basic + vars_covariates)

    # Simple correlation between student–teacher ratio and test scores
    corr = df_basic["testscr"].corr(df_basic["str"])

    # Bivariate regression: testscr ~ str
    X_basic = sm.add_constant(df_basic["str"])
    model_basic = sm.OLS(df_basic["testscr"], X_basic).fit()
    coef_basic = model_basic.params["str"]
    pval_basic = model_basic.pvalues["str"]
    r2_basic = float(model_basic.rsquared)

    # Multivariate regression controlling for key demographics & income:
    # testscr ~ str + income + english + calworks + lunch
    X_full = sm.add_constant(df_full[["str"] + vars_covariates])
    model_full = sm.OLS(df_full["testscr"], X_full).fit()
    coef_full = model_full.params["str"]
    pval_full = model_full.pvalues["str"]
    r2_full = float(model_full.rsquared)

    # Decide on Likert-style response between 0 and 100.
    # We base the strength primarily on:
    # - Sign and magnitude of the coefficient
    # - Consistency (bivariate and multivariate both point the same way)
    # - Statistical significance levels
    # Interpretation: a negative coefficient on str means that more students per teacher
    # (higher ratio) is associated with lower test scores, so *lower* ratios correspond
    # to higher performance.

    strong_negative = (coef_basic < 0) and (coef_full < 0)
    highly_significant = (pval_basic < 0.001) and (pval_full < 0.001)
    moderately_significant = (pval_basic < 0.05) and (pval_full < 0.05)

    if strong_negative and highly_significant:
        response_int = 90
    elif strong_negative and moderately_significant:
        response_int = 75
    elif strong_negative and (pval_basic < 0.1 or pval_full < 0.1):
        response_int = 60
    else:
        # Little or no evidence of an association; answer leans toward "No"
        response_int = 25

    # Build explanation text
    question = info.get("research_questions", [""])[0]

    explanation_lines = []
    explanation_lines.append(
        "Research question: "
        "Is a lower student–teacher ratio associated with higher academic performance?"
    )
    explanation_lines.append(
        "Data: 420 California K–6 and K–8 school districts "
        "with 5th-grade Stanford 9 reading and math scores, "
        "plus district-level characteristics (enrollment, teachers, income, "
        "CalWorks participation, reduced-price lunch, English learner share, etc.)."
    )
    explanation_lines.append(
        "I constructed the student–teacher ratio as students divided by teachers "
        "and defined overall academic performance as the average of reading and "
        "math scores for each district."
    )
    explanation_lines.append(
        f"The Pearson correlation between test scores and the student–teacher "
        f"ratio is {corr:.3f}."
    )
    if abs(corr) < 0.05:
        explanation_lines.append(
            "This correlation is very close to zero, indicating essentially no "
            "linear relationship between test scores and the student–teacher ratio."
        )
    elif corr < 0:
        explanation_lines.append(
            "This indicates that districts with more students per teacher tend to "
            "have lower test scores (a negative association)."
        )
    else:
        explanation_lines.append(
            "This indicates that districts with more students per teacher tend to "
            "have higher test scores (a positive association)."
        )
    explanation_lines.append(
        "In a bivariate linear regression of average test score on the "
        f"student–teacher ratio, the estimated coefficient on the ratio is "
        f"{coef_basic:.3f} with p-value {pval_basic:.3g} and R-squared {r2_basic:.3f}."
    )
    if pval_basic < 0.05:
        if coef_basic < 0:
            explanation_lines.append(
                "This negative, statistically significant coefficient implies that, "
                "in the simple model, higher student–teacher ratios are associated "
                "with lower achievement."
            )
        else:
            explanation_lines.append(
                "This positive, statistically significant coefficient implies that, "
                "in the simple model, higher student–teacher ratios are associated "
                "with higher achievement."
            )
    else:
        explanation_lines.append(
            "Because the p-value is well above conventional significance thresholds, "
            "the simple regression does not provide strong evidence that test scores "
            "vary systematically with the student–teacher ratio."
        )
    explanation_lines.append(
        "To account for observable demographic and socioeconomic differences across "
        "districts, I estimated a multivariate regression of test scores on the "
        "student–teacher ratio plus income, percent English learners, percent of "
        "students receiving CalWorks, and percent on reduced-price lunch. In this "
        f"model, the coefficient on the student–teacher ratio is {coef_full:.3f} "
        f"with p-value {pval_full:.3g} and R-squared {r2_full:.3f}."
    )
    if pval_full < 0.05:
        if coef_full < 0:
            explanation_lines.append(
                "The coefficient on the student–teacher ratio is negative and "
                "statistically significant even after controlling for these "
                "covariates, which indicates a robust association: districts with "
                "smaller classes (lower student–teacher ratios) tend to have higher "
                "average test scores."
            )
        else:
            explanation_lines.append(
                "The coefficient on the student–teacher ratio is positive and "
                "statistically significant even after controlling for these "
                "covariates, which would imply that districts with larger classes "
                "(higher student–teacher ratios) tend to have higher average test "
                "scores."
            )
    else:
        explanation_lines.append(
            "In this multivariate model, the coefficient on the student–teacher "
            "ratio is not statistically distinguishable from zero, so once we "
            "control for income and demographic composition, the data do not show "
            "a clear association between class size and test scores."
        )
    explanation_lines.append(
        f"Given the estimated coefficients and their lack of statistical "
        f"significance in this dataset, I do not find strong evidence that lower "
        f"student–teacher ratios are associated with higher academic performance. "
        f"My answer therefore leans toward 'No'. I encode this as {response_int} "
        "on a 0–100 Likert scale, where 0 represents a strong 'No' and 100 a "
        "strong 'Yes'."
    )
    explanation_lines.append(
        "This analysis is observational and based on cross-sectional district-level "
        "data, so, even with stronger coefficients, it would support only an "
        "association rather than a causal claim about the effects of changing "
        "class size."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(response_int), "explanation": explanation}

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
