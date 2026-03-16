import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    n = len(df)
    str_mean = float(df["stratio"].mean())
    testscr_mean = float(df["testscr"].mean())
    testscr_sd = float(df["testscr"].std(ddof=1))

    # Simple regression: test score on student–teacher ratio
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    beta_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key demographic and resource controls
    controls = ["income", "english", "lunch"]
    X_ctrl = sm.add_constant(df[["stratio"] + controls])
    model_ctrl = sm.OLS(df["testscr"], X_ctrl).fit()
    beta_ctrl = float(model_ctrl.params["stratio"])
    p_ctrl = float(model_ctrl.pvalues["stratio"])
    r2_ctrl = float(model_ctrl.rsquared)

    # Effect size: change in test score (in SD units) per 1-student change in ratio
    if testscr_sd > 0:
        effect_sd = abs(beta_ctrl) / testscr_sd
    else:
        effect_sd = 0.0

    # Map statistical evidence to a 0–100 Likert response
    if p_ctrl < 0.05 and beta_ctrl < 0:
        # Evidence that lower student–teacher ratios are associated with higher scores
        if p_ctrl < 0.001 and effect_sd >= 0.25:
            response = 88
        elif effect_sd >= 0.15:
            response = 78
        else:
            response = 68
    elif p_ctrl < 0.05 and beta_ctrl > 0:
        # Significant association in the opposite direction
        if effect_sd >= 0.25:
            response = 12
        else:
            response = 25
    else:
        # No statistically convincing association after controls
        if beta_ctrl < 0:
            response = 45
        else:
            response = 40

    # Translate coefficient into an interpretable difference for a 5-student change
    delta_5 = -5.0 * beta_ctrl  # reduction in ratio by 5 students

    strength_desc = "strong" if response >= 80 else "moderate" if response >= 60 else "weak"
    direction_desc = (
        "lower student–teacher ratios are associated with higher test scores"
        if beta_ctrl < 0
        else "there is no clear evidence that lower student–teacher ratios are associated with higher test scores"
    )

    explanation = (
        f"Using data on {n} California K-6 and K-8 districts, I constructed the student–teacher ratio "
        f"(students divided by teachers, mean {str_mean:.1f}) and an overall academic performance measure "
        f"(the average of reading and math Stanford 9 scores, mean {testscr_mean:.1f}). "
        f"In a simple linear regression of average test score on the student–teacher ratio, each additional student "
        f"per teacher is associated with {beta_simple:.2f} points change in average test score "
        f"(p = {p_simple:.3g}, R² = {r2_simple:.3f}). "
        f"Adding controls for district income, the percentage of English learners, and the percentage of students "
        f"eligible for subsidized lunch, the estimated effect of the student–teacher ratio remains "
        f"{beta_ctrl:.2f} points in test score per additional student per teacher (p = {p_ctrl:.3g}, "
        f"R² = {r2_ctrl:.3f}). "
        f"Given the observed standard deviation of test scores ({testscr_sd:.2f}), this corresponds to an effect size "
        f"of about {effect_sd:.2f} standard deviations for a one-student change in the ratio, so a 5-student reduction "
        f"in the student–teacher ratio is associated with roughly {delta_5:.1f} points higher average test scores "
        f"holding these demographic factors constant. "
        f"Overall, I find that {direction_desc}: the association is statistically "
        f"{'significant' if p_ctrl < 0.05 else 'not statistically significant at conventional levels'} "
        f"after adjusting for key confounders. "
        f"The Likert-scale response value of {response} therefore reflects a {strength_desc} Yes answer to the "
        f"question of whether lower student–teacher ratios are associated with higher academic performance, while "
        f"recognizing that these are observational correlations rather than causal estimates."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
