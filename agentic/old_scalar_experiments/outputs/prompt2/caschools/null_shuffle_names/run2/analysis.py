import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(meta_path: Path) -> dict:
    with meta_path.open("r") as f:
        return json.load(f)


def find_column_by_description(fields, phrase: str) -> str:
    phrase_lower = phrase.lower()
    for field in fields:
        desc = field.get("properties", {}).get("description", "") or ""
        if phrase_lower in desc.lower():
            return field["column"]
    raise ValueError(f"No column found with description containing: {phrase}")


def main() -> None:
    base_dir = Path(__file__).parent

    meta = load_metadata(base_dir / "info.json")
    fields = meta["data_desc"]["fields"]

    # Identify key semantic columns from descriptions, independent of shuffled names.
    enrollment_col = find_column_by_description(fields, "Total enrollment")
    teachers_col = find_column_by_description(fields, "Number of teachers")
    readscore_col = find_column_by_description(fields, "Average reading score")
    mathscore_col = find_column_by_description(fields, "Average math score")

    income_col = find_column_by_description(fields, "District average income")
    calworks_pct_col = find_column_by_description(
        fields, "Percent qualifying for CalWorks"
    )
    lunch_pct_col = find_column_by_description(
        fields, "Percent qualifying for reduced-price lunch"
    )
    english_learner_pct_col = find_column_by_description(
        fields, "Percent of English learners"
    )
    expend_per_student_col = find_column_by_description(
        fields, "Expenditure per student"
    )

    df = pd.read_csv(base_dir / "caschools.csv")

    # Construct key variables.
    df = df.copy()
    df["enrollment"] = df[enrollment_col].astype(float)
    df["teachers"] = df[teachers_col].astype(float)

    # Guard against invalid or zero teacher counts when computing ratios.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["teachers"] > 0].copy()

    df["stratio"] = df["enrollment"] / df["teachers"]

    df["readscore"] = df[readscore_col].astype(float)
    df["mathscore"] = df[mathscore_col].astype(float)
    df["testscr"] = (df["readscore"] + df["mathscore"]) / 2.0

    # Basic association: Pearson correlation between student-teacher ratio and test scores.
    corr = df["testscr"].corr(df["stratio"])

    # Simple bivariate regression.
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple, missing="drop").fit()
    beta_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    # Multiple regression controlling for key socioeconomic and resource covariates.
    covariates = [
        "stratio",
        income_col,
        calworks_pct_col,
        lunch_pct_col,
        english_learner_pct_col,
        expend_per_student_col,
    ]

    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["testscr"], X_multi, missing="drop").fit()
    beta_multi = model_multi.params["stratio"]
    pval_multi = model_multi.pvalues["stratio"]

    # Decide on "Yes"/"No" based on the sign and statistical strength of association.
    # We interpret "associated" as a reasonably robust negative relationship.
    is_negative = (beta_simple < 0) and (beta_multi < 0)
    is_stat_sig = (pval_simple < 0.05) and (pval_multi < 0.05)

    if is_negative and is_stat_sig:
        response = "Yes"
        confidence = 90
    elif is_negative and (pval_simple < 0.1 or pval_multi < 0.1):
        response = "Yes"
        confidence = 70
    else:
        response = "No"
        confidence = 60 if not is_negative else 55

    # Build explanation string summarizing the evidence.
    explanation_lines = [
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?",
        f"I constructed a student-teacher ratio as total enrollment ({enrollment_col}) divided by number of teachers ({teachers_col}), and an overall test score as the average of the reading ({readscore_col}) and math ({mathscore_col}) scores.",
        f"The Pearson correlation between the student-teacher ratio and the composite test score is {corr:.3f}, where a negative value indicates that fewer students per teacher is associated with higher scores.",
        f"A simple OLS regression of test scores on the student-teacher ratio yields a coefficient of {beta_simple:.3f} (p-value = {pval_simple:.3g}).",
        f"An OLS regression controlling for income, poverty proxies (CalWorks and reduced-price lunch), percent English learners, and per-student expenditure yields a coefficient on the student-teacher ratio of {beta_multi:.3f} (p-value = {pval_multi:.3g}).",
        "Based on the sign and statistical significance of these coefficients, I assessed whether there is consistent evidence that districts with lower student-teacher ratios tend to have higher academic performance.",
    ]

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    # Write required JSON object to conclusion.txt with no extra text.
    out_path = base_dir / "conclusion.txt"
    with out_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

