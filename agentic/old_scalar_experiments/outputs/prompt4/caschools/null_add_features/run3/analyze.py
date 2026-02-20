import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio and an overall test score
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any missing values just in case
    df = df[["stratio", "testscr"]].dropna()

    # Correlation between class size and performance
    corr = df["stratio"].corr(df["testscr"])

    # Simple OLS regression: test score on student–teacher ratio
    X = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model = sm.OLS(y, X).fit()

    coef = float(model.params["stratio"])
    pval = float(model.pvalues["stratio"])
    r2 = float(model.rsquared)

    # Map statistical evidence to a 0–100 Likert-style response.
    # Higher values correspond to stronger evidence that
    # lower student–teacher ratios are associated with higher performance.
    abs_corr = abs(corr)

    if coef < 0:
        # Evidence in the expected direction
        if pval < 0.001 and abs_corr >= 0.3:
            response = 90
            strength = "strong"
        elif pval < 0.05 and abs_corr >= 0.1:
            response = 80
            strength = "moderate"
        elif pval < 0.1:
            response = 65
            strength = "weak"
        else:
            response = 55
            strength = "very weak"
    else:
        # Either no effect or effect opposite to the hypothesis
        if pval < 0.05 and abs_corr >= 0.1:
            response = 20
            strength = "moderate"
        elif pval < 0.1:
            response = 35
            strength = "weak"
        else:
            response = 45
            strength = "very weak"

    # Build a concise textual explanation
    explanation = (
        "Using data from 420 California K-6 and K-8 school districts, "
        "I computed the student–teacher ratio as students divided by teachers "
        "and defined overall academic performance as the average of the reading "
        "and math test scores. "
        f"The Pearson correlation between student–teacher ratio and average test score "
        f"is {corr:.3f}, indicating a {('negative' if corr < 0 else 'positive')} "
        "association. "
        f"A simple OLS regression of average test score on student–teacher ratio yields "
        f"a slope coefficient of {coef:.3f} with p-value {pval:.3g} and R-squared {r2:.3f}. "
        "This means that, holding other factors constant in this simple model, "
        "an additional student per teacher is associated with an estimated change "
        "in average test score equal to the slope coefficient. "
        f"Because the estimated effect is {('negative' if coef < 0 else 'positive')} "
        f"and the statistical evidence is {strength} (based on the p-value and "
        "correlation magnitude), "
        "the data provide "
        + (
            "meaningful support"
            if response >= 70
            else "only limited support"
            if 55 <= response < 70
            else "little or no support"
        )
        + " for the claim that lower student–teacher ratios are associated with higher "
        "academic performance."
    )

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

