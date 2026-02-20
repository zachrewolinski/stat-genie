import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Define student–teacher ratio and overall test score
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key fields if present
    key_cols = ["stratio", "testscr", "calworks", "lunch", "income", "english"]
    df_model = df.dropna(subset=key_cols)

    # Simple Pearson correlation between ratio and test scores
    corr = float(df_model["stratio"].corr(df_model["testscr"]))

    # Multivariate linear regression controlling for observed covariates
    y = df_model["testscr"]
    X = df_model[["stratio", "calworks", "lunch", "income", "english"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    coef_stratio = float(model.params["stratio"])
    pvalue_stratio = float(model.pvalues["stratio"])
    r_squared = float(model.rsquared)

    # Decide on answer based on sign and significance of the association
    if coef_stratio < 0 and pvalue_stratio < 0.05:
        response = "Yes"
        # Map p-value to a heuristic confidence score, capped for conservatism
        if pvalue_stratio < 0.001:
            confidence = 92
        elif pvalue_stratio < 0.01:
            confidence = 88
        else:
            confidence = 80
        association_case = "negative_significant"
    elif coef_stratio > 0 and pvalue_stratio < 0.05:
        # Statistically significant association in the opposite direction
        response = "No"
        confidence = 88
        association_case = "positive_significant"
    else:
        # No statistically clear association
        response = "No"
        if pvalue_stratio < 0.1:
            confidence = 65
        else:
            confidence = 55
        association_case = "no_clear"

    # Build explanation text that matches the numerical results
    explanation_parts = [
        "Using data from 420 California K-6 and K-8 school districts, "
        "I computed the student–teacher ratio as students divided by teachers and "
        "overall academic performance as the average of 5th-grade reading and math test scores. ",
        f"The simple Pearson correlation between student–teacher ratio and test scores is {corr:.3f}, ",
    ]

    if abs(corr) < 0.05:
        explanation_parts.append(
            "which is very close to zero and suggests little linear association between class size and performance. "
        )
    elif corr < 0:
        explanation_parts.append(
            "indicating that districts with more students per teacher tend to have lower scores. "
        )
    else:
        explanation_parts.append(
            "indicating that districts with more students per teacher tend to have higher scores. "
        )

    explanation_parts.append(
        "To account for observable demographic and economic differences, I ran a linear regression of "
        "test scores on the student–teacher ratio while controlling for the percentages of students on "
        "CalWorks and reduced-price lunch, average district income, and the percentage of English learners. "
        f"In this regression, the coefficient on the student–teacher ratio is {coef_stratio:.3f} with a "
        f"p-value of {pvalue_stratio:.4f} and model R-squared of {r_squared:.3f}. "
    )

    if association_case == "negative_significant":
        explanation_parts.append(
            "The negative and statistically significant coefficient means that, within this dataset, districts "
            "with more students per teacher have lower test scores, so lower student–teacher ratios (smaller classes) "
            "are associated with higher academic performance even after adjusting for these covariates. "
        )
    elif association_case == "positive_significant":
        explanation_parts.append(
            "The positive and statistically significant coefficient means that, within this dataset, districts "
            "with more students per teacher actually have higher test scores; this pattern runs counter to the "
            "hypothesis that smaller classes are associated with higher performance. "
        )
    else:
        explanation_parts.append(
            "The coefficient is very small in magnitude and not statistically distinguishable from zero, so after "
            "adjusting for these covariates the data do not show a clear association between class size and academic "
            "performance in either direction. "
        )

    explanation_parts.append(
        "Because the data are observational and limited to one state and time period, this establishes association "
        "at most and not definitive causation, so I avoid over-interpreting the estimates and focus on the direction "
        "and strength of the relationships observed."
    )

    explanation = "".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
