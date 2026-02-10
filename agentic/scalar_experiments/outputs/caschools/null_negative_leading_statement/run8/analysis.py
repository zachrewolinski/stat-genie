import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def run_analysis() -> int:
    info = load_metadata(Path("info.json"))
    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Model 1: bivariate relationship between class size and performance
    X1 = sm.add_constant(df[["stratio"]])
    y = df["avg_score"]
    model1 = sm.OLS(y, X1).fit()
    coef1 = float(model1.params["stratio"])
    pval1 = float(model1.pvalues["stratio"])

    # Model 2: control for key socioeconomic and resource covariates
    controls = df[["income", "english", "lunch", "calworks", "expenditure"]]
    X2 = sm.add_constant(pd.concat([df["stratio"], controls], axis=1))
    model2 = sm.OLS(y, X2).fit()
    coef2 = float(model2.params["stratio"])
    pval2 = float(model2.pvalues["stratio"])

    # Prefer the controlled model for inference, fall back to simple if needed
    coef_use = coef2
    p_use = pval2
    if np.isnan(coef_use) or np.isnan(p_use):
        coef_use = coef1
        p_use = pval1

    # Map evidence to Likert-scale integer in [-100, 100]
    if np.isnan(coef_use) or np.isnan(p_use):
        scalar = 0
    elif p_use >= 0.1:
        scalar = 0
    else:
        if p_use < 0.01:
            base = 80
        elif p_use < 0.05:
            base = 60
        else:
            base = 30
        scalar = base if coef_use < 0 else -base

    scalar = int(max(-100, min(100, scalar)))

    # Save a brief human-readable summary of the analysis
    summary_lines = [
        f"Research question: {research_question}",
        "",
        "Bivariate model (avg_score ~ stratio):",
        f"  coef(stratio) = {coef1:.4f}, p = {pval1:.4g}",
        "",
        "Controlled model (avg_score ~ stratio + income + english + lunch + calworks + expenditure):",
        f"  coef(stratio) = {coef2:.4f}, p = {pval2:.4g}",
        "",
        f"Derived Likert scalar (−100 to 100): {scalar}",
    ]

    if scalar > 0:
        interpretation = (
            "Evidence that lower student-teacher ratios are associated with "
            "higher academic performance (negative stratio coefficient)."
        )
    elif scalar < 0:
        interpretation = (
            "Evidence against a beneficial association of lower student-teacher "
            "ratios with performance (positive or negligible effect)."
        )
    else:
        interpretation = (
            "Little to no statistically reliable association between class size "
            "and academic performance."
        )

    summary_lines.append("")
    summary_lines.append(f"Interpretation: {interpretation}")

    Path("analysis_summary.txt").write_text("\n".join(summary_lines))

    return scalar


def main() -> None:
    scalar = run_analysis()
    # Per instructions, conclusion.txt must contain only the scalar integer
    Path("conclusion.txt").write_text(str(scalar))


if __name__ == "__main__":
    main()

