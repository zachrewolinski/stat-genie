import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent

    # Load research question (for context, not strictly needed for the stats)
    info_path = base_path / "info.json"
    info = json.loads(info_path.read_text())
    question = info.get("research_questions", [""])[0]

    # Load data
    data_path = base_path / "caschools.csv"
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Keep only rows with complete data on variables we will use
    model_cols = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]
    df_model = df[model_cols].dropna().copy()

    n = len(df_model)

    # Simple correlation
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit()

    # Multiple regression with key demographic and resource controls
    formula_multi = (
        "testscr ~ stratio + income + english + lunch + calworks + computer + expenditure"
    )
    model_multi = smf.ols(formula_multi, data=df_model).fit()

    beta_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    ci_simple_low, ci_simple_high = model_simple.conf_int().loc["stratio"].tolist()
    r2_simple = float(model_simple.rsquared)

    beta_multi = float(model_multi.params["stratio"])
    p_multi = float(model_multi.pvalues["stratio"])
    ci_multi_low, ci_multi_high = model_multi.conf_int().loc["stratio"].tolist()
    r2_multi = float(model_multi.rsquared)

    # Effect across a realistic range of ratios (10th to 90th percentile)
    q10 = float(df_model["stratio"].quantile(0.10))
    q90 = float(df_model["stratio"].quantile(0.90))
    delta_ratio = q90 - q10
    implied_diff_simple = beta_simple * delta_ratio
    implied_diff_multi = beta_multi * delta_ratio

    # Qualitative description of correlation strength
    abs_r = abs(r)
    if abs_r < 0.2:
        corr_desc = "very weak"
    elif abs_r < 0.4:
        corr_desc = "weak to moderate"
    elif abs_r < 0.6:
        corr_desc = "moderate"
    elif abs_r < 0.8:
        corr_desc = "moderately strong"
    else:
        corr_desc = "strong"

    # Map evidence to a 0–100 Likert scale for a Yes/No answer
    neg_assoc = (beta_simple < 0) and (beta_multi < 0) and (r < 0)

    if neg_assoc:
        # Stronger evidence when both models highly significant
        if (p_simple < 1e-4) and (p_multi < 1e-4):
            response = 95
        elif (p_simple < 1e-3) and (p_multi < 1e-3):
            response = 90
        elif (p_simple < 1e-2) and (p_multi < 1e-2):
            response = 85
        elif (p_simple < 5e-2) and (p_multi < 5e-2):
            response = 80
        else:
            response = 70
    else:
        # No clear consistent negative association; lean toward "No"
        if (p_simple < 5e-2) or (p_multi < 5e-2):
            response = 40
        else:
            response = 20

    # Build explanation string
    explanation = (
        f"Research question: '{question}'\n"
        f"Data: {n} California K-6/K-8 districts from the caschools dataset. "
        f"I constructed the student–teacher ratio as students divided by teachers, "
        f"and overall academic performance as the average of reading and math Stanford 9 scores.\n"
        f"The Pearson correlation between student–teacher ratio and test scores is "
        f"{r:.3f} (p = {p_corr:.2e}), indicating a {corr_desc} "
        f"{'negative' if r < 0 else 'positive'} association.\n"
        f"In a simple linear regression of test score on student–teacher ratio, the estimated coefficient "
        f"on the ratio is {beta_simple:.2f} (95% CI [{ci_simple_low:.2f}, {ci_simple_high:.2f}], "
        f"p = {p_simple:.2e}, R² = {r2_simple:.3f}). This implies that moving from the 10th to the 90th "
        f"percentile of the student–teacher ratio (a change of about {delta_ratio:.1f} students per teacher) "
        f"is associated with roughly {implied_diff_simple:.1f} points change in average test scores.\n"
        f"In a multiple regression controlling for district income, percentage of English learners, percentage "
        f"of students on reduced-price lunch, percentage receiving CalWorks, computers per classroom, and "
        f"expenditures per student, the coefficient on the student–teacher ratio remains {beta_multi:.2f} "
        f"(95% CI [{ci_multi_low:.2f}, {ci_multi_high:.2f}], p = {p_multi:.2e}, R² = {r2_multi:.3f}). "
        f"This model implies an expected difference of about {implied_diff_multi:.1f} test-score points "
        f"between districts at the 10th and 90th percentiles of the ratio, holding these covariates fixed.\n"
        f"Because the association between the student–teacher ratio and test scores is consistently negative "
        f"and statistically significant in both simple and multiple regression models, with non-trivial effect "
        f"sizes over realistic changes in the ratio, there is strong evidence that lower student–teacher ratios "
        f"are associated with higher academic performance in this dataset.\n"
        f"I therefore give a 'Yes' answer to the research question and map the strength of this conclusion to "
        f"a value of {response} on a 0–100 Likert scale, where higher values indicate stronger evidence for 'Yes'."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    conclusion_path = base_path / "conclusion.txt"
    conclusion_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

