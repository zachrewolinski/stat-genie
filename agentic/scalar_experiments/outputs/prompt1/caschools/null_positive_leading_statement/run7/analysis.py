import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio and average test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key variables
    key_vars = [
        "stratio",
        "testscr",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
    ]
    available_key_vars = [c for c in key_vars if c in df.columns]
    df = df.dropna(subset=available_key_vars)

    # Correlation between student-teacher ratio and test scores
    corr = df["testscr"].corr(df["stratio"])

    # Simple OLS regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    beta_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]

    # Multiple OLS regression with controls, if available
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    available_controls = [c for c in controls if c in df.columns]

    beta_multi = None
    p_multi = None
    model_multi = None

    if available_controls:
        X_multi = sm.add_constant(df[["stratio"] + available_controls])
        model_multi = sm.OLS(df["testscr"], X_multi).fit()
        beta_multi = model_multi.params["stratio"]
        p_multi = model_multi.pvalues["stratio"]

    # Decide on binary response based on sign and significance
    response = "No"
    evidence_for = False

    if beta_simple < 0 and p_simple < 0.05:
        if beta_multi is None or (beta_multi < 0 and p_multi < 0.05):
            response = "Yes"
            evidence_for = True

    # Build explanation text
    explanation_parts = []
    explanation_parts.append(
        f"We examined the relationship between the student–teacher ratio "
        f"(students per teacher) and academic performance, measured as the "
        f"average of reading and math scores, across {len(df)} California K-6 "
        f"and K-8 school districts."
    )
    explanation_parts.append(
        f"The Pearson correlation between student–teacher ratio and average "
        f"test scores is {corr:.3f}, indicating that districts with "
        f"{'higher' if corr > 0 else 'lower'} student–teacher ratios tend to "
        f"have {'higher' if corr > 0 else 'lower'} scores."
    )
    explanation_parts.append(
        f"In a simple OLS regression of average test scores on student–teacher "
        f"ratio, the estimated coefficient on the ratio is {beta_simple:.3f} "
        f"(p-value = {p_simple:.3g})."
    )

    if model_multi is not None:
        explanation_parts.append(
            "We also estimated a multiple regression controlling for "
            f"socioeconomic and demographic factors "
            f"({', '.join(available_controls)}). In this model, the "
            f"coefficient on student–teacher ratio is {beta_multi:.3f} "
            f"(p-value = {p_multi:.3g})."
        )

    if evidence_for:
        explanation_parts.append(
            "Both analyses show a statistically significant negative "
            "association: as the student–teacher ratio decreases (smaller "
            "classes), average test scores increase, including when "
            "controlling for observed covariates where available."
        )
    else:
        explanation_parts.append(
            "These analyses do not provide strong, statistically significant "
            "evidence that lower student–teacher ratios are associated with "
            "higher test scores once sampling variability and observed "
            "covariates are taken into account."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    # Write JSON conclusion to file with no extra text
    Path("conclusion.txt").write_text(
        json.dumps(conclusion, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

