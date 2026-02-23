import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic descriptive statistics
    corr = df["stratio"].corr(df["testscr"])

    # Simple linear regression: average test score on student-teacher ratio
    model_simple = smf.ols("testscr ~ stratio", data=df).fit(cov_type="HC3")

    # Multiple regression controlling for observable demographics and resources
    formula_controls = "testscr ~ stratio + income + english + lunch + calworks + expenditure + computer"
    model_controls = smf.ols(formula_controls, data=df).fit(cov_type="HC3")

    print("Number of districts:", len(df))
    print("Correlation between student-teacher ratio and test score:", corr)
    print("\nSimple regression (HC3 robust SE):")
    print(model_simple.summary())
    print("\nRegression with controls (HC3 robust SE):")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

