import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Simple linear regression: test score on student-teacher ratio
    X = sm.add_constant(df[["stratio"]])
    y = df["testscr"]
    model = sm.OLS(y, X).fit()

    coef_str = float(model.params["stratio"])
    pval_str = float(model.pvalues["stratio"])

    # Standardized effect size: effect for 1 SD change in ratio
    str_sd = float(df["stratio"].std())
    testscr_sd = float(df["testscr"].std())
    if testscr_sd > 0:
        effect_std = abs(coef_str * str_sd / testscr_sd)
    else:
        effect_std = 0.0

    # Map evidence strength to a Likert-style scalar in [-100, 100]
    if pval_str < 0.001:
        conf_factor = 1.0
    elif pval_str < 0.01:
        conf_factor = 0.75
    elif pval_str < 0.05:
        conf_factor = 0.5
    elif pval_str < 0.1:
        conf_factor = 0.25
    else:
        conf_factor = 0.0

    if effect_std >= 0.5:
        effect_factor = 1.0
    elif effect_std >= 0.3:
        effect_factor = 0.75
    elif effect_std >= 0.1:
        effect_factor = 0.5
    elif effect_std >= 0.05:
        effect_factor = 0.25
    else:
        effect_factor = 0.0

    base_score = int(round(100 * conf_factor * effect_factor))

    if coef_str < 0:
        # Lower student-teacher ratio (smaller classes) associated with higher scores
        scalar = base_score
    elif coef_str > 0:
        # Higher student-teacher ratio associated with higher scores (opposite of question)
        scalar = -base_score
    else:
        scalar = 0

    # Ensure scalar is within [-100, 100]
    scalar = max(-100, min(100, int(scalar)))

    # Print brief diagnostic output for human review
    print("Coefficient on student-teacher ratio:", coef_str)
    print("p-value:", pval_str)
    print("Standardized effect size:", effect_std)
    print("Likert-scale scalar ([-100, 100]):", scalar)

    # Write the scalar conclusion to the required file
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

