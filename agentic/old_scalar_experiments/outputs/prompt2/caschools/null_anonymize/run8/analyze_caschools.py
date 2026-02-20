import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    info_path = base_dir / "info.json"
    data_path = base_dir / "caschools.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Construct key derived variables
    df = df.copy()
    # Student-teacher ratio: enrollment / number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]
    # Academic performance: average of reading and math scores
    df["avgscore"] = (df["feature14"] + df["feature15"]) / 2.0

    # Clean data: drop problematic rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "avgscore"])

    stratio = df["stratio"]
    avgscore = df["avgscore"]

    # Correlation between student-teacher ratio and academic performance
    r, p_corr = stats.pearsonr(stratio, avgscore)

    # Simple linear regression: avgscore ~ stratio
    X_simple = sm.add_constant(stratio)
    model_simple = sm.OLS(avgscore, X_simple).fit()
    slope_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    # Multiple regression with key controls
    control_candidates = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    controls = [c for c in control_candidates if c in df.columns]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(avgscore, X_controls).fit()
    slope_ctrl = float(model_controls.params["stratio"])
    p_ctrl = float(model_controls.pvalues["stratio"])

    # Determine binary answer based on direction and significance of association
    if slope_simple < 0 and p_simple < 0.05 and slope_ctrl < 0 and p_ctrl < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Confidence score based on strength and robustness of evidence
    if response == "Yes":
        if abs(r) >= 0.4 and p_ctrl < 0.01:
            confidence = 90
        elif abs(r) >= 0.2 and p_ctrl < 0.05:
            confidence = 80
        else:
            confidence = 65
    else:
        if abs(r) <= 0.1 or (slope_simple * slope_ctrl > 0 and (p_simple > 0.1 or p_ctrl > 0.1)):
            confidence = 70
        else:
            confidence = 60

    question = info.get("research_questions", [""])[0]
    direction = "lower" if r < 0 else "higher"
    perf_dir = "higher" if r < 0 else "lower"

    explanation = (
        f"Research question: {question} "
        f"Using the California school districts data, I computed the student–teacher ratio as total enrollment "
        f"divided by the number of teachers and defined academic performance as the average of reading and math scores. "
        f"The Pearson correlation between student–teacher ratio and average score is {r:.3f} "
        f"(p = {p_corr:.3g}), indicating that districts with {direction} student–teacher ratios tend to have "
        f"{perf_dir} test scores. A simple linear regression of average score on the ratio yields a slope of "
        f"{slope_simple:.3f} (p = {p_simple:.3g}). A multiple regression controlling for economic disadvantage, "
        f"English learners, and spending (CalWorks %, reduced-price lunch %, expenditures, income, and English-learner %) "
        f"yields a slope of {slope_ctrl:.3f} (p = {p_ctrl:.3g}). Together, these results "
        f"{'provide evidence for' if response == 'Yes' else 'do not provide strong evidence for'} "
        f"a relationship in this dataset where lower student–teacher ratios are associated with higher academic performance."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

