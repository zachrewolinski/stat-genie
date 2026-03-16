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

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "stratio",
            "testscr",
            "income",
            "english",
            "lunch",
            "calworks",
            "expenditure",
            "computer",
        ]
    )

    # Correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(df["stratio"], df["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    slope_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression with controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(y, X_controls).fit()
    slope_ctrl = model_controls.params["stratio"]
    p_ctrl = model_controls.pvalues["stratio"]
    r2_ctrl = model_controls.rsquared

    testscr_sd = df["testscr"].std()

    print("Research question:", research_question)
    print()
    print("Number of observations:", len(df))
    print()
    print("Correlation between student-teacher ratio and test scores:")
    print(f"  r = {r:.3f}, p-value = {p_corr:.4g}")
    print()
    print("Simple OLS: testscr ~ stratio")
    print(f"  slope (stratio) = {slope_simple:.3f}")
    print(f"  p-value (stratio) = {p_simple:.4g}")
    print(f"  R^2 = {r2_simple:.3f}")
    print()
    print("OLS with controls: testscr ~ stratio + income + english + lunch + calworks + expenditure + computer")
    print(f"  slope (stratio) = {slope_ctrl:.3f}")
    print(f"  p-value (stratio) = {p_ctrl:.4g}")
    print(f"  R^2 = {r2_ctrl:.3f}")
    print()
    print("Standard deviation of test scores:", f"{testscr_sd:.3f}")
    if testscr_sd > 0:
        effect_size = slope_ctrl / testscr_sd
        print("Effect size (controls model):", f"{effect_size:.3f}")
    else:
        print("Effect size (controls model): NA")


if __name__ == "__main__":
    main()
