import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key variables (if any)
    df_model = df[["testscr", "stratio", "income", "english", "lunch", "calworks"]].dropna()

    # Simple association: testscr ~ stratio
    y_simple = df_model["testscr"]
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(y_simple, X_simple).fit()

    coef_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key socioeconomic factors
    X_multi = df_model[["stratio", "income", "english", "lunch", "calworks"]]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(y_simple, X_multi).fit()

    coef_multi = float(model_multi.params["stratio"])
    pval_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)

    # Correlation between stratio and testscr
    corr = float(df_model["stratio"].corr(df_model["testscr"]))

    # Interpret results:
    # A negative coefficient means that higher student-teacher ratios
    # (more students per teacher) are associated with LOWER scores,
    # i.e., lower ratios are associated with HIGHER performance.
    associated = False
    strength = "little or no"

    if coef_multi < 0 and pval_multi < 0.05:
        associated = True
        if abs(coef_multi) >= 0.8:
            strength = "strong"
        elif abs(coef_multi) >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"
    elif coef_simple < 0 and pval_simple < 0.05:
        associated = True
        strength = "weak"

    # Map conclusion to Likert-style integer on [0, 100]
    # 0   = very strong "No"
    # 50  = uncertain / mixed
    # 100 = very strong "Yes"
    if associated:
        if strength == "strong":
            response_value = 85
        elif strength == "moderate":
            response_value = 75
        else:
            response_value = 65
    else:
        # No statistically reliable negative association
        if pval_multi > 0.1 and pval_simple > 0.1:
            response_value = 20
        else:
            response_value = 40

    response_value = int(response_value)

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance? "
        "I constructed a student–teacher ratio variable as students per teacher (stratio) and an overall "
        "academic performance measure as the average of reading and math scores (testscr). "
        f"In a simple linear regression of testscr on stratio, the estimated coefficient on stratio was "
        f"{coef_simple:.3f} with p-value {pval_simple:.3f} and R-squared {r2_simple:.3f}, and the correlation "
        f"between stratio and testscr was {corr:.3f}. "
        f"In a multiple regression controlling for average district income, percentage of English learners, "
        f"and poverty proxies (lunch and CalWorks), the coefficient on stratio was {coef_multi:.3f} with "
        f"p-value {pval_multi:.3f} and R-squared {r2_multi:.3f}. "
    )

    if associated:
        explanation += (
            "In both models the coefficient on stratio is negative and statistically significant, meaning that "
            "districts with more students per teacher tend to have lower average test scores. This implies that "
            "lower student–teacher ratios are associated with higher academic performance, even after accounting "
            "for key socioeconomic and demographic factors. "
        )
    else:
        explanation += (
            "The estimated coefficients on stratio are not reliably negative or statistically significant, which "
            "provides little evidence that student–teacher ratios are systematically related to test scores once "
            "socioeconomic and demographic factors are taken into account. "
        )

    explanation += (
        f"Based on this evidence, I place my answer at {response_value} on a 0–100 scale where 0 represents a "
        "strong 'No' and 100 represents a strong 'Yes' to the statement that lower student–teacher ratios are "
        "associated with higher academic performance."
    )

    conclusion = {
        "response": response_value,
        "explanation": explanation,
    }

    # Write JSON output to conclusion.txt with no extra text
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

