import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Core variables based on metadata and research question
    # Student–teacher ratio
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance: mean of reading and math scores
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key variables, if any
    core_cols = ["avg_score", "stratio", "income", "english", "calworks", "lunch", "computer", "expenditure"]
    df_model = df.dropna(subset=core_cols).copy()

    # Simple correlation between student–teacher ratio and average score
    r_simple = df_model["avg_score"].corr(df_model["stratio"])

    # Simple linear regression: avg_score ~ stratio
    X_simple = sm.add_constant(df_model[["stratio"]])
    y = df_model["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "english", "calworks", "lunch", "computer", "expenditure"]
    X_multi = sm.add_constant(df_model[["stratio"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()
    coef_multi = model_multi.params["stratio"]
    pval_multi = model_multi.pvalues["stratio"]
    r2_multi = model_multi.rsquared

    # Decide on Yes/No and Likert score (0–100)
    # Lower ratio (smaller stratio) associated with higher performance corresponds to
    # a negative coefficient/correlation.
    is_negative_simple = coef_simple < 0
    is_negative_multi = coef_multi < 0
    significant_simple = pval_simple < 0.05
    significant_multi = pval_multi < 0.05

    if is_negative_simple and is_negative_multi and significant_simple and significant_multi:
        # Consistently negative and statistically significant association
        # Scale strength by standardized effect size (rough heuristic using correlation).
        strength = min(1.0, max(0.0, abs(r_simple)))
        base_score = 70  # clear evidence of association
        response_value = int(round(base_score + 20 * strength))
        yes_no_text = "YES"
    elif (is_negative_simple and significant_simple) or (is_negative_multi and significant_multi):
        # Some evidence of negative association but not perfectly consistent
        strength = min(1.0, max(0.0, abs(r_simple)))
        base_score = 55
        response_value = int(round(base_score + 15 * strength))
        yes_no_text = "YES (moderate evidence)"
    else:
        # Little to no consistent evidence that lower ratios are associated with higher performance.
        strength = min(1.0, max(0.0, abs(r_simple)))
        base_score = 30
        response_value = int(round(base_score - 20 * (1.0 - strength)))
        response_value = max(0, response_value)
        yes_no_text = "NO"

    response_value = int(max(0, min(100, response_value)))

    if significant_simple and significant_multi:
        evidence_text = "statistically significant in both simple and adjusted models"
    else:
        evidence_text = "weaker and/or not consistently statistically significant across models"

    explanation_lines = [
        "Research question: Is a lower student–teacher ratio associated with higher academic performance?",
        "",
        "Key setup:",
        "- Dataset: 420 California K-6/K-8 districts (caschools.csv).",
        "- Student–teacher ratio computed as students / teachers (stratio).",
        "- Academic performance measured as the average of reading and math test scores (avg_score).",
        "",
        "Simple association:",
        f"- Pearson correlation between avg_score and stratio: r = {r_simple:.3f} (negative implies lower ratios -> higher scores).",
        f"- Simple OLS regression avg_score ~ stratio: coefficient for stratio = {coef_simple:.3f}, p-value = {pval_simple:.4f}, R^2 = {r2_simple:.3f}.",
        "",
        "Adjusted association (controlling for income, English learner share, CalWorks, reduced-price lunch, computers per student, and expenditures):",
        f"- Multiple OLS regression avg_score ~ stratio + controls: coefficient for stratio = {coef_multi:.3f}, p-value = {pval_multi:.4f}, R^2 = {r2_multi:.3f}.",
        "",
        f"Interpretation: The estimated effect of the student–teacher ratio is "
        f"{'negative' if coef_multi < 0 else 'not consistently negative'}, and the evidence of a relationship is {evidence_text}.",
        "",
        f"Conclusion: {yes_no_text}. On a 0–100 Likert scale, I assign a value of {response_value}, where higher values indicate stronger evidence that lower student–teacher ratios are associated with higher academic performance.",
    ]

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": response_value,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
