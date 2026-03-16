import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).resolve().parent

    info = json.loads((base_path / "info.json").read_text())
    research_question = info["research_questions"][0]

    df = pd.read_csv(base_path / "caschools.csv")

    # Map anonymized feature names to meaningful variables based on info.json descriptions.
    # feature6: total enrollment (students), feature7: number of teachers,
    # feature14: average reading score, feature15: average math score.
    df = df.copy()
    df["students"] = df["feature6"]
    df["teachers"] = df["feature7"]
    df["read"] = df["feature14"]
    df["math"] = df["feature15"]

    # Construct key derived variables.
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obviously problematic rows (e.g. zero or missing teachers).
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["str", "testscr"])

    # Simple correlation as a first pass.
    corr = df["str"].corr(df["testscr"])

    # Bivariate linear regression: testscr on str.
    X = sm.add_constant(df["str"])
    y = df["testscr"]
    model_biv = sm.OLS(y, X).fit()

    slope = float(model_biv.params["str"])
    p_value = float(model_biv.pvalues["str"])
    r_squared = float(model_biv.rsquared)

    # Interpret: lower str (smaller classes) vs testscr.
    # Negative slope means lower str -> higher testscr.
    if p_value < 0.01 and slope < 0:
        likert = 90
    elif p_value < 0.05 and slope < 0:
        likert = 75
    elif p_value < 0.1 and slope < 0:
        likert = 60
    elif p_value >= 0.1 and slope < 0:
        likert = 40
    else:
        # No evidence of the hypothesized negative relationship.
        likert = 20

    direction = "negative" if slope < 0 else "positive"

    explanation_lines = [
        f"Research question: {research_question}",
        "Operationalization:",
        "- Student–teacher ratio (STR) = total enrollment / number of teachers.",
        "- Academic performance = mean of average reading and math scores.",
        "",
        "Evidence:",
        f"- Pearson correlation between STR and test scores: {corr:.3f}.",
        f"- Bivariate OLS of test scores on STR: slope = {slope:.3f}, p-value = {p_value:.4g}, R^2 = {r_squared:.3f}.",
        f"- The estimated association is {direction}: higher STR (larger classes) is associated with "
        f"{'lower' if slope < 0 else 'higher'} test scores.",
        "",
        "Conclusion:",
        "Based on the size and statistical significance of the estimated relationship, "
        "I translate the strength of evidence that lower student–teacher ratios are associated "
        "with higher academic performance into the provided Likert scale.",
    ]

    explanation = "\n".join(explanation_lines)

    conclusion = {"response": likert, "explanation": explanation}

    (base_path / "conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

