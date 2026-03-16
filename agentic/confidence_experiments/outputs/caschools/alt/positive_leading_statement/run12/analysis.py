import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(".")

    info_path = base_path / "info.json"
    data_path = base_path / "caschools.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic descriptive statistics
    desc = df[["stratio", "testscr"]].describe()

    corr = df[["stratio", "testscr"]].corr().loc["stratio", "testscr"]

    # Simple bivariate regression
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()

    # Regression with key demographic and resource controls
    model_controls = smf.ols(
        "testscr ~ stratio + english + lunch + calworks + income + expenditure",
        data=df,
    ).fit()

    # Alternative specification with a slightly simpler set of controls
    model_controls_simple = smf.ols(
        "testscr ~ stratio + english + lunch + income", data=df
    ).fit()

    # Collect key results for quick inspection
    print("Research question:")
    print(question)
    print()

    print("Descriptive statistics for student-teacher ratio and test scores:")
    print(desc)
    print()

    print(f"Correlation between student-teacher ratio and test scores: {corr:.3f}")
    print()

    def coef_info(model, name: str) -> None:
        coef = model.params.get("stratio", np.nan)
        pval = model.pvalues.get("stratio", np.nan)
        print(f"{name}: coef(stratio) = {coef:.3f}, p-value = {pval:.4f}")

    coef_info(model_simple, "Simple OLS")
    coef_info(model_controls, "OLS with controls")
    coef_info(model_controls_simple, "OLS with simpler controls")
    print()

    print("Simple OLS summary (truncated):")
    print(model_simple.summary())
    print()

    print("OLS with controls summary (truncated):")
    print(model_controls.summary())
    print()

    print("OLS with simpler controls summary (truncated):")
    print(model_controls_simple.summary())


if __name__ == "__main__":
    main()
