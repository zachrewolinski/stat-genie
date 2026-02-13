import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


INFO_PATH = Path("info.json")
DATA_PATH = Path("caschools.csv")
OUTPUT_PATH = Path("conclusion.txt")


def load_variable_mapping() -> dict:
    """Infer semantic variable roles from info.json descriptions."""
    with INFO_PATH.open() as f:
        info = json.load(f)

    mapping: dict = {}
    for field in info.get("data_desc", {}).get("fields", []):
        col = field.get("column")
        props = field.get("properties", {}) or {}
        desc = (props.get("description") or "").lower()

        if not desc:
            continue

        if "total enrollment" in desc:
            mapping["enrollment"] = col
        if "number of teachers" in desc:
            mapping["teachers"] = col
        if "average reading score" in desc:
            mapping["read_score"] = col
        if "average math score" in desc:
            mapping["math_score"] = col
        if "district average income" in desc:
            mapping["income"] = col
        if "percent of english learners" in desc:
            mapping["ell_pct"] = col
        if "percent qualifying for calworks" in desc:
            mapping["calworks_pct"] = col
        if "percent qualifying for reduced-price lunch" in desc:
            mapping["lunch_pct"] = col
        if "expenditure per student" in desc:
            mapping["exp_per_student"] = col

    return mapping


def main() -> None:
    mapping = load_variable_mapping()

    enrollment_col = mapping.get("enrollment")
    teachers_col = mapping.get("teachers")
    read_col = mapping.get("read_score")
    math_col = mapping.get("math_score")

    # Basic sanity checks to avoid silently using wrong columns.
    for key, col in [
        ("enrollment", enrollment_col),
        ("teachers", teachers_col),
        ("reading score", read_col),
        ("math score", math_col),
    ]:
        if col is None:
            raise RuntimeError(f"Could not identify column for {key} from metadata.")

    df = pd.read_csv(DATA_PATH)
    df = df.copy()

    # Student–teacher ratio: total enrollment divided by number of teachers.
    df["str"] = df[enrollment_col] / df[teachers_col]

    # Academic performance: mean of reading and math scores.
    df["testscr"] = df[[read_col, math_col]].mean(axis=1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["str", "testscr"])

    # Simple correlation and regression of test scores on STR.
    corr = df["str"].corr(df["testscr"])

    x_simple = sm.add_constant(df["str"])
    model_simple = sm.OLS(df["testscr"], x_simple).fit()
    slope_simple = model_simple.params["str"]
    p_simple = model_simple.pvalues["str"]
    r2_simple = model_simple.rsquared

    # Multiple regression with available covariates, if any.
    covariate_keys = [
        "income",
        "ell_pct",
        "calworks_pct",
        "lunch_pct",
        "exp_per_student",
    ]
    cov_cols = [mapping[k] for k in covariate_keys if mapping.get(k) in df.columns]

    slope_multi = None
    p_multi = None
    r2_multi = None

    if cov_cols:
        x_multi = sm.add_constant(df[["str"] + cov_cols])
        model_multi = sm.OLS(df["testscr"], x_multi).fit()
        slope_multi = model_multi.params["str"]
        p_multi = model_multi.pvalues["str"]
        r2_multi = model_multi.rsquared

    # Determine binary answer.
    negative_association = slope_simple < 0
    statistically_significant = p_simple < 0.05

    if negative_association and statistically_significant:
        response = "Yes"
    else:
        response = "No"

    # Confidence heuristic based on strength and robustness of evidence.
    confidence = 50
    if negative_association:
        confidence += 15
    if statistically_significant:
        confidence += 20
    if abs(corr) > 0.2:
        confidence += 5
    if abs(corr) > 0.3:
        confidence += 5
    if slope_multi is not None and slope_multi < 0 and (p_multi is not None and p_multi < 0.05):
        confidence += 5

    confidence = max(0, min(100, int(round(confidence))))

    # Build explanation text summarizing methods and key statistics.
    lines = []
    lines.append(
        "I examined whether a lower student–teacher ratio is associated with higher "
        "academic performance by computing the student–teacher ratio as total enrollment "
        "divided by the number of teachers and defining academic performance as the mean "
        "of district reading and math scores."
    )
    lines.append(
        f"The Pearson correlation between the student–teacher ratio and average test scores "
        f"was {corr:.3f}. A simple linear regression of test scores on the student–teacher "
        f"ratio yielded a slope of {slope_simple:.3f} score points per additional student "
        f"per teacher (p = {p_simple:.3g}, R² = {r2_simple:.3f})."
    )
    if slope_multi is not None and p_multi is not None and r2_multi is not None:
        lines.append(
            "I also estimated a multiple regression controlling for district income, the share "
            "of English learners, poverty proxies, and expenditures per student where available; "
            f"in this model the coefficient on the student–teacher ratio was {slope_multi:.3f} "
            f"(p = {p_multi:.3g}, R² = {r2_multi:.3f})."
        )
    if response == "Yes":
        lines.append(
            "Because the estimated association is negative and statistically significant, these "
            "results indicate that districts with lower student–teacher ratios tend to have "
            "higher average test scores in this dataset. This reflects an observational "
            "association and should not be interpreted as a definitive causal effect."
        )
    else:
        lines.append(
            "Given the estimated effect sizes and statistical uncertainty, the data do not "
            "provide clear evidence that districts with lower student–teacher ratios have "
            "higher average test scores."
        )

    explanation = " ".join(lines)

    output = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

