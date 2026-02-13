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

    # Basic summaries
    corr = df["stratio"].corr(df["testscr"])

    # Simple bivariate regression: testscr on stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression with controls for demographics and resources
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()

    print("Number of districts:", len(df))
    print("Mean student-teacher ratio:", df["stratio"].mean())
    print("Correlation between stratio and testscr:", corr)
    print()
    print("Simple regression: testscr ~ stratio")
    print(model_simple.summary())
    print()
    print("Regression with controls: testscr ~ stratio + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

