import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = pathlib.Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Core variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Simple Pearson correlation between student–teacher ratio and test score
    corr = df["stratio"].corr(df["testscr"])

    # Regression of test scores on student–teacher ratio, controlling for key covariates
    covariates = ["calworks", "lunch", "english", "income", "expenditure"]
    model_df = df[["testscr", "stratio"] + covariates].dropna()

    X = model_df[["stratio"] + covariates]
    X = sm.add_constant(X)
    y = model_df["testscr"]

    model = sm.OLS(y, X).fit()
    coef_str = model.params["stratio"]
    t_str = model.tvalues["stratio"]

    # Map evidence strength to a Likert-style scalar in [-100, 100]
    # The research question is directional: lower STR -> higher performance.
    # A negative coefficient supports a "Yes" answer.
    sign = -1 if coef_str < 0 else 1

    # Use both effect size (per-student change) and statistical strength (t-stat)
    # to derive a confidence score, capped at 1.0 in magnitude.
    effect_scale = min(abs(coef_str) / 2.0, 1.0)  # typical CASchools effect is well below 2
    t_scale = min(abs(t_str) / 5.0, 1.0)  # t around 5+ is already very strong evidence
    confidence = min(effect_scale * 0.4 + t_scale * 0.6, 1.0)

    scalar = int(round(sign * confidence * 100))

    # Print a brief analysis summary for inspection in the CLI.
    print("Correlation (STR vs testscr):", float(corr))
    print("OLS coef for STR:", float(coef_str))
    print("t-stat for STR:", float(t_str))
    print("Derived Likert scalar:", scalar)

    # Write the required scalar conclusion file with ONLY the integer value.
    conclusion_path = pathlib.Path("conclusion.txt")
    conclusion_path.write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

