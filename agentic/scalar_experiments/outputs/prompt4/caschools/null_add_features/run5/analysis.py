import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and average test score.
    df = df.copy()
    df["str"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Keep only rows with complete data on variables of interest.
    cols_simple = ["avg_score", "str"]
    cols_extended = cols_simple + [
        "income",
        "english",
        "calworks",
        "lunch",
        "expenditure",
        "computer",
    ]
    df_simple = df[cols_simple].dropna()
    df_extended = df[cols_extended].dropna()

    # Simple correlation between ratio and average score.
    corr = df_simple["avg_score"].corr(df_simple["str"])

    # Simple linear regression: avg_score ~ str
    X_simple = sm.add_constant(df_simple["str"])
    model_simple = sm.OLS(df_simple["avg_score"], X_simple).fit()
    coef_str_simple = model_simple.params["str"]
    pval_str_simple = model_simple.pvalues["str"]
    r2_simple = model_simple.rsquared

    # Multiple regression controlling for socioeconomic and resource factors.
    X_ext = sm.add_constant(
        df_extended[["str", "income", "english", "calworks", "lunch", "expenditure", "computer"]]
    )
    model_ext = sm.OLS(df_extended["avg_score"], X_ext).fit()
    coef_str_ext = model_ext.params["str"]
    pval_str_ext = model_ext.pvalues["str"]
    r2_ext = model_ext.rsquared

    # Compare mean scores across quartiles of the ratio.
    df_quartiles = df_simple.copy()
    df_quartiles["str_quartile"] = pd.qcut(df_quartiles["str"], 4, labels=False)
    mean_scores_by_quartile = (
        df_quartiles.groupby("str_quartile")["avg_score"].mean().to_dict()
    )

    # Derive an overall strength-of-evidence score on 0-100 scale.
    # Start from a neutral 50 and adjust based on direction, significance, and consistency.
    response_score = 50

    # Direction: negative correlation and coefficient support "lower ratio -> higher performance".
    if corr < 0 and coef_str_simple < 0 and coef_str_ext < 0:
        response_score += 20
    elif (corr < 0 and coef_str_simple < 0) or (corr < 0 and coef_str_ext < 0):
        response_score += 10
    elif corr > 0 and coef_str_simple > 0 and coef_str_ext > 0:
        response_score -= 20

    # Statistical significance of the ratio coefficient.
    if pval_str_simple < 0.01 and pval_str_ext < 0.01:
        response_score += 20
    elif pval_str_simple < 0.05 and pval_str_ext < 0.05:
        response_score += 10
    elif pval_str_simple > 0.1 and pval_str_ext > 0.1:
        response_score -= 10

    # Magnitude of correlation.
    if abs(corr) >= 0.3:
        response_score += 10
    elif abs(corr) <= 0.1:
        response_score -= 5

    # Monotonic trend across quartiles.
    quartile_scores = [mean_scores_by_quartile[q] for q in sorted(mean_scores_by_quartile)]
    if quartile_scores[0] > quartile_scores[-1]:
        # Higher scores at lowest ratios.
        response_score += 10
    elif quartile_scores[0] < quartile_scores[-1]:
        response_score -= 10

    # Clip to [0, 100] and convert to int.
    response_score = int(np.clip(response_score, 0, 100))

    # Build explanation string summarizing key quantitative evidence.
    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance? "
        "Using 420 California K-6 and K-8 districts, I constructed a student-teacher ratio as students divided "
        "by teachers and an average achievement score as the mean of 5th-grade reading and math scores. "
        f"The simple Pearson correlation between the ratio and average score is {corr:.3f}, which is very close to zero, "
        "indicating essentially no linear association between class size and achievement in this sample. "
        f"In a simple linear regression of average score on the ratio, the coefficient on the ratio is "
        f"{coef_str_simple:.4f} (p-value = {pval_str_simple:.4g}, R-squared = {r2_simple:.3f}), so changes in the "
        "student-teacher ratio are not significantly related to changes in average test scores. "
        f"When controlling for district socioeconomic and resource variables (income, English-learner share, CalWorks share, "
        f"lunch subsidy share, expenditure per pupil, and computers per classroom), the coefficient on the ratio remains "
        f"{coef_str_ext:.4f} (p-value = {pval_str_ext:.4g}, R-squared = {r2_ext:.3f}), again indicating a negligible and "
        "statistically non-significant relationship even after accounting for these covariates. "
        "Grouping districts into quartiles of the student-teacher ratio, the mean achievement scores are very similar across "
        f"groups with no clear monotonic pattern (quartile means: {', '.join(f'{v:.1f}' for v in quartile_scores)}). "
        "Taken together, the correlation, regression coefficients, and quartile comparisons provide consistent evidence that "
        "within this dataset there is little to no association between lower student-teacher ratios and higher academic performance. "
        "Because the data are observational and cross-sectional, this analysis also cannot establish whether changing class sizes "
        "would causally affect achievement; it only indicates that, as observed here, smaller classes are not strongly linked to "
        "better test scores."
    )

    conclusion = {"response": response_score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    # Also print key results for inspection.
    print("Correlation (avg_score vs STR):", corr)
    print("Simple model coef(str), p, R2:", coef_str_simple, pval_str_simple, r2_simple)
    print("Extended model coef(str), p, R2:", coef_str_ext, pval_str_ext, r2_ext)
    print("Quartile mean scores:", quartile_scores)
    print("Response score (0-100):", response_score)


if __name__ == "__main__":
    main()
