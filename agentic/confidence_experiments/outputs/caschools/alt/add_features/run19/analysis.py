import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    base = Path(__file__).parent

    # Load research question (for context in the explanation)
    info = json.loads((base / "info.json").read_text())
    question = info.get("research_questions", [""])[0]

    # Load dataset
    df = pd.read_csv(base / "caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df_analysis = df.dropna(subset=["stratio", "testscr"]).copy()

    # Basic summaries
    n = df_analysis.shape[0]
    str_mean = df_analysis["stratio"].mean()
    str_sd = df_analysis["stratio"].std()
    ts_mean = df_analysis["testscr"].mean()
    ts_sd = df_analysis["testscr"].std()

    # Correlation between class size and performance
    r, p_corr = stats.pearsonr(df_analysis["stratio"], df_analysis["testscr"])
    direction_phrase = (
        "districts with smaller classes tend to have higher scores"
        if r < 0
        else "districts with larger classes tend to have higher scores"
    )

    # Simple OLS: testscr ~ stratio
    X1 = sm.add_constant(df_analysis["stratio"])
    y1 = df_analysis["testscr"]
    model1 = sm.OLS(y1, X1).fit()
    coef1 = float(model1.params["stratio"])
    p1 = float(model1.pvalues["stratio"])
    r2_1 = float(model1.rsquared)

    # OLS with covariates
    control_candidates = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    controls = [c for c in control_candidates if c in df_analysis.columns]

    if controls:
        df_reg2 = df_analysis.dropna(subset=controls).copy()
        X2 = sm.add_constant(df_reg2[["stratio"] + controls])
        y2 = df_reg2["testscr"]
        model2 = sm.OLS(y2, X2).fit()
        coef2 = float(model2.params["stratio"])
        p2 = float(model2.pvalues["stratio"])
        r2_2 = float(model2.rsquared)
        n2 = df_reg2.shape[0]
    else:
        df_reg2 = df_analysis
        coef2 = coef1
        p2 = p1
        r2_2 = r2_1
        n2 = n

    # Determine strength of evidence that lower ratios are associated with higher performance
    # (i.e., we expect a negative coefficient for stratio).
    negative_association_simple = coef1 < 0
    negative_association_adjusted = coef2 < 0

    # Heuristic mapping from evidence to 0-100 scale
    if negative_association_adjusted and p2 < 0.001 and abs(r) >= 0.30:
        response = 90
        strength_desc = "strong evidence of a meaningful negative association"
    elif negative_association_adjusted and p2 < 0.01 and abs(r) >= 0.20:
        response = 80
        strength_desc = "clear and moderately strong negative association"
    elif negative_association_adjusted and p2 < 0.05:
        response = 70
        strength_desc = "statistically significant but modest negative association"
    elif negative_association_simple and (p1 < 0.05 or p_corr < 0.05):
        response = 55
        strength_desc = "some evidence of a negative association, but not robust after adjustment"
    else:
        response = 25
        strength_desc = "little robust evidence that lower ratios are associated with higher performance"

    yes_no = "Yes" if response >= 50 else "No"

    # Build narrative explanation
    explanation_lines = [
        f"Research question: {question}",
        f"Data: {n} California K–6/K–8 districts with 5th grade Stanford 9 test scores.",
        f"Student–teacher ratio (students per teacher) has mean {str_mean:.2f} (SD {str_sd:.2f});",
        f"average test score (mean of reading and math) has mean {ts_mean:.1f} (SD {ts_sd:.1f}).",
        "",
        "Bivariate association:",
        f"- Pearson correlation between student–teacher ratio and average test score is r = {r:.3f} (p = {p_corr:.3g}), indicating that {direction_phrase}.",
        "",
        "Regression analysis:",
        f"- Simple OLS regression testscr ~ stratio yields coefficient {coef1:.3f} (p = {p1:.3g}, R² = {r2_1:.3f}).",
        (
            f"  Interpreting the sign, a one-student increase in the student–teacher ratio is associated with "
            f"{abs(coef1):.2f}-point {'lower' if coef1 < 0 else 'higher'} average test scores in this simple model."
        ),
        "",
        "Adjusted for covariates:",
    ]

    if controls:
        explanation_lines.append(
            f"- Including controls for {', '.join(controls)}, the adjusted model (n = {n2}) "
            f"gives a coefficient on student–teacher ratio of {coef2:.3f} (p = {p2:.3g}, R² = {r2_2:.3f})."
        )
        explanation_lines.append(
            "  This shows how much of the association between class size and performance remains after "
            "accounting for socio-economic and demographic differences across districts."
        )
    else:
        explanation_lines.append(
            "- No additional covariates were available beyond the student–teacher ratio."
        )

    explanation_lines.extend(
        [
            "",
            "Overall assessment:",
            f"- The pattern of results indicates {strength_desc}.",
            f"- Based on this, I answer '{yes_no}' to the question of whether lower student–teacher ratios are",
            f"  associated with higher academic performance, assigning a strength of {response} on a 0–100 Likert scale.",
        ]
    )

    explanation = "\n".join(explanation_lines)

    conclusion = {"response": int(response), "explanation": explanation}
    (base / "conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

