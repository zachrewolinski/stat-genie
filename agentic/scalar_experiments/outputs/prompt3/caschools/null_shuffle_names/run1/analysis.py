import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the descriptions in info.json to map columns to their semantic meaning.

    english   -> total enrollment
    students  -> number of teachers
    district  -> average reading score
    expenditure -> average math score
    school    -> percent CalWorks
    computer  -> percent reduced-price lunch
    grades    -> expenditure per student
    income    -> district average income (in $1,000)
    rownames  -> percent English learners
    """
    df = df.copy()

    # Student–teacher ratio: students per teacher.
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: average of reading and math scores.
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Demographic and resource controls.
    df["calworks_pct"] = df["school"]
    df["lunch_pct"] = df["computer"]
    df["el_pct"] = df["rownames"]
    df["expn_stu"] = df["grades"]
    df["income_k"] = df["income"]

    # Drop rows with missing or infinite values in key fields.
    key_cols = [
        "stratio",
        "testscr",
        "income_k",
        "el_pct",
        "lunch_pct",
        "calworks_pct",
        "expn_stu",
    ]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=key_cols)
    return df


def run_analysis(df: pd.DataFrame) -> dict:
    """
    Quantify the association between student–teacher ratio and test scores.
    """
    # Simple Pearson correlation.
    corr = float(df["stratio"].corr(df["testscr"]))

    # Simple OLS: testscr ~ stratio.
    y = df["testscr"]
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])

    # Multiple regression with key controls.
    controls = ["income_k", "el_pct", "lunch_pct", "calworks_pct", "expn_stu"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()
    coef_multi = float(model_multi.params["stratio"])
    pval_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)

    n = int(df.shape[0])

    # Decide directional Yes/No based on sign and significance.
    # Lower ratios correspond to higher performance if increasing ratio
    # (more students per teacher) is associated with lower scores,
    # i.e., coef for stratio is negative.
    effect_consistent = (coef_simple < 0) and (coef_multi < 0)
    significant = (pval_simple < 0.05) and (pval_multi < 0.05)

    if effect_consistent and significant:
        response = "Yes"
    else:
        response = "No"

    # Strength of the Yes/No: based primarily on |correlation|.
    strength = int(round(min(100.0, max(0.0, abs(corr) * 100.0))))

    # Confidence heuristic based on p-values and sample size.
    if not effect_consistent:
        base_conf = 40.0
    elif pval_multi < 1e-6:
        base_conf = 90.0
    elif pval_multi < 1e-3:
        base_conf = 80.0
    elif pval_multi < 0.01:
        base_conf = 70.0
    elif pval_multi < 0.05:
        base_conf = 60.0
    else:
        base_conf = 45.0

    # Slightly up-weight confidence for larger samples.
    size_boost = min(5.0, max(0.0, (n - 100) / 100.0))
    confidence = int(round(max(0.0, min(100.0, base_conf + size_boost))))

    explanation_parts = [
        f"The analysis used data from {n} school districts.",
        "Student–teacher ratio was computed as total enrollment divided by the number of teachers.",
        "Academic performance was measured as the average of district reading and math scores.",
        f"The Pearson correlation between student–teacher ratio and test scores was {corr:.3f},",
        f"and the simple OLS regression coefficient on the ratio was {coef_simple:.3f} (p = {pval_simple:.3g}).",
        "A multiple regression including income, percent English learners, percent reduced-price lunch,",
        f"percent CalWorks, and expenditure per student yielded a coefficient of {coef_multi:.3f} on the ratio",
        f"(p = {pval_multi:.3g}, R-squared = {r2_multi:.3f}).",
    ]

    if response == "Yes":
        explanation_parts.append(
            "Because higher student–teacher ratios are associated with lower test scores, "
            "the results indicate that lower student–teacher ratios are associated with higher academic performance."
        )
    else:
        explanation_parts.append(
            "The estimated relationship between student–teacher ratios and test scores was not consistently negative "
            "and statistically significant, so the data do not provide strong evidence that lower ratios are "
            "associated with higher academic performance."
        )

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    df = load_data(Path("caschools.csv"))
    df = construct_variables(df)
    result = run_analysis(df)

    # Ensure output directory is current working directory.
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

