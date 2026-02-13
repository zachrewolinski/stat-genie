import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio
    df["stratio"] = df["students"] / df["teachers"]

    # Outcome: average of reading and math scores
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables, just in case
    df_model = df[["score", "stratio"]].dropna()

    X = sm.add_constant(df_model["stratio"])
    y = df_model["score"]

    model = sm.OLS(y, X).fit()

    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]
    r2 = model.rsquared

    # Determine answer: lower STR associated with higher performance
    # corresponds to a negative coefficient on stratio.
    if coef < 0 and pval < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Confidence heuristic
    if pval < 0.001 and coef < 0:
        confidence = 95
    elif pval < 0.05 and coef < 0:
        confidence = 85
    elif pval < 0.1 and coef < 0:
        confidence = 70
    else:
        confidence = 60

    explanation = (
        "I regressed average test scores (mean of reading and math) on the "
        "student–teacher ratio using ordinary least squares. The estimated "
        f"coefficient on the student–teacher ratio is {coef:.3f} with p-value "
        f"{pval:.3g} and R-squared {r2:.3f}. A negative and statistically "
        "significant coefficient indicates that districts with fewer students "
        "per teacher tend to have higher test scores, controlling only for the "
        "intercept."
    )

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()

