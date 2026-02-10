import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing core variables, if any
    core_cols = ["stratio", "testscr", "income", "english", "lunch", "calworks", "expenditure"]
    df_model = df.dropna(subset=core_cols).copy()

    # Simple correlation between student-teacher ratio and test scores
    r = df_model["testscr"].corr(df_model["stratio"])

    # Linear model with basic controls to estimate partial effect
    X = df_model[["stratio", "income", "english", "lunch", "calworks", "expenditure"]]
    X = sm.add_constant(X)
    y = df_model["testscr"]

    model = sm.OLS(y, X).fit()
    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]

    # Standardized effect size for the ratio coefficient
    std_effect = abs(coef * df_model["stratio"].std() / df_model["testscr"].std())

    # Map evidence to Likert-style scalar in [-100, 100]
    # Sign is oriented so positive values support the hypothesis
    # that lower student-teacher ratios are associated with higher performance.
    sign = -1.0 * np.sign(coef)  # negative coef => positive support

    # p-value component: 0 -> 1, 0.5 -> 0
    p_component = 1.0 - min(pval, 0.5) / 0.5

    # Effect-size component: cap at 1 for reasonably large standardized effects
    effect_component = min(std_effect / 0.5, 1.0)

    # Correlation component: uses magnitude of simple correlation
    corr_component = min(abs(r) / 0.5, 1.0)

    # Combine components (average) and scale to [-100, 100]
    evidence_strength = (p_component + effect_component + corr_component) / 3.0
    scalar = int(round(sign * evidence_strength * 100))

    # Ensure scalar lies within [-100, 100]
    scalar = max(-100, min(100, scalar))

    # Write final scalar to conclusion.txt with no extra text or lines
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

