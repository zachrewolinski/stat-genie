import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student–teacher ratio and overall test score.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables we will use.
    model_cols = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
    ]
    df_model = df[model_cols].dropna()

    # Simple correlation between student–teacher ratio and test scores.
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    coef_str_simple = model_simple.params["stratio"]
    pval_str_simple = model_simple.pvalues["stratio"]

    # Multiple regression controlling for key demographics and resources.
    X_controls = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(y, X_controls).fit()
    coef_str_ctrl = model_controls.params["stratio"]
    pval_str_ctrl = model_controls.pvalues["stratio"]
    r2_ctrl = model_controls.rsquared

    # Decide on answer scale:
    # Negative coefficient means lower ratio (smaller classes) is associated with higher scores.
    # We encode confidence based on sign and significance from the controlled model.
    if coef_str_ctrl < 0 and pval_str_ctrl < 0.01:
        response = 80
    elif coef_str_ctrl < 0 and pval_str_ctrl < 0.05:
        response = 70
    elif coef_str_ctrl < 0 and pval_str_ctrl < 0.10:
        response = 60
    elif coef_str_ctrl < 0:
        response = 55
    elif abs(coef_str_ctrl) <= 0.1 or pval_str_ctrl > 0.5:
        response = 50
    else:
        # Positive and at least somewhat meaningful effect.
        response = 30

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "(here measured by the average of reading and math scores).\n\n"
        f"I constructed a student–teacher ratio as students/teachers (mean {df_model['stratio'].mean():.1f}, "
        f"SD {df_model['stratio'].std():.1f}) and an overall test score as the mean of reading and math scores "
        f"(mean {df_model['testscr'].mean():.1f}, SD {df_model['testscr'].std():.1f}) across {len(df_model)} districts.\n\n"
        f"The Pearson correlation between student–teacher ratio and test scores is {corr:.3f}. "
        "A negative value means that smaller classes (lower ratios) are associated with higher scores.\n\n"
        f"In a simple linear regression of test scores on the student–teacher ratio, the coefficient on the ratio "
        f"is {coef_str_simple:.3f} with p-value {pval_str_simple:.3f}. "
        "This indicates how test scores change on average when the ratio increases by one student per teacher.\n\n"
        "To account for observable differences between districts, I then estimated a multiple regression of test scores "
        "on the student–teacher ratio while controlling for income, the shares of students in CalWorks, qualifying for "
        "reduced-price lunch, and who are English learners, as well as expenditure per student. "
        f"In this model (R² = {r2_ctrl:.3f}), the coefficient on the student–teacher ratio is {coef_str_ctrl:.3f} "
        f"with p-value {pval_str_ctrl:.3f}.\n\n"
        "Because the coefficient on the student–teacher ratio in the controlled model is "
        f"{'negative' if coef_str_ctrl < 0 else 'positive'} and "
        f"{'statistically significant' if pval_str_ctrl < 0.05 else 'not strongly statistically significant'} "
        "at conventional levels, the evidence "
        f"{'supports' if coef_str_ctrl < 0 else 'does not support'} the claim that smaller student–teacher ratios "
        "are associated with higher academic performance, though the magnitude of the association is modest in size. "
        "The reported response score reflects both the sign and statistical strength of this estimated relationship."
    )

    conclusion = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

