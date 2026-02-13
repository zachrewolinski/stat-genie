import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Construct key variables
    df["str_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key fields, if present
    df = df.dropna(subset=["str_ratio", "avg_score", "income", "english", "lunch", "calworks"])

    # Simple correlation
    corr_coef, corr_p = stats.pearsonr(df["str_ratio"], df["avg_score"])

    # Regression models
    model_simple = smf.ols("avg_score ~ str_ratio", data=df).fit()
    model_controls = smf.ols(
        "avg_score ~ str_ratio + income + english + lunch + calworks", data=df
    ).fit()

    coef_simple = model_simple.params["str_ratio"]
    pval_simple = model_simple.pvalues["str_ratio"]

    coef_controls = model_controls.params["str_ratio"]
    pval_controls = model_controls.pvalues["str_ratio"]

    # Decision logic: is lower student-teacher ratio associated with higher performance?
    # This corresponds to a negative association between str_ratio and avg_score.
    negative_and_significant = (
        corr_coef < 0
        and corr_p < 0.05
        and coef_simple < 0
        and pval_simple < 0.05
        and coef_controls < 0
        and pval_controls < 0.05
    )

    mostly_negative = coef_simple < 0 and coef_controls < 0
    some_significance = (pval_simple < 0.1) or (pval_controls < 0.1) or (corr_p < 0.1)

    if negative_and_significant:
        response = "Yes"
        confidence = 90
    elif mostly_negative and some_significance:
        response = "Yes"
        confidence = 75
    else:
        response = "No"
        confidence = 60

    # Build explanation with key numerical evidence
    explanation = (
        "I examined the association between the student–teacher ratio and average test scores "
        "(mean of reading and math) across all districts. The simple Pearson correlation between "
        "the student–teacher ratio and average score was "
        f"{corr_coef:.3f} (p-value {corr_p:.4f}). In a simple linear regression of average score "
        f"on the student–teacher ratio, the coefficient on the ratio was {coef_simple:.3f} "
        f"with p-value {pval_simple:.4f}. In a multiple regression that additionally controlled "
        f"for district income, percent English learners, percent on CalWorks, and percent on "
        f"reduced-price lunch, the coefficient on the student–teacher ratio was {coef_controls:.3f} "
        f"with p-value {pval_controls:.4f}. These results indicate that "
        "a lower student–teacher ratio is associated with higher academic performance "
        "whenever the estimated relationship is negative and statistically meaningful; "
        "the conclusion and confidence score above summarize the strength and consistency "
        "of this evidence."
    )

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

