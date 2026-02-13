import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_PATH = Path("caschools.csv")


def load_and_engineer() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Student-teacher ratio: students per teacher
    df["stratio"] = df["feature6"] / df["feature7"]
    # Overall test score: average of reading and math
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0
    return df


def simple_association(df: pd.DataFrame) -> dict:
    out: dict = {}
    corr = df["stratio"].corr(df["testscr"])
    out["corr_stratio_testscr"] = float(corr)

    # Simple OLS: testscr ~ stratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X, missing="drop").fit()
    out["ols_coef_stratio"] = float(model.params["stratio"])
    out["ols_pvalue_stratio"] = float(model.pvalues["stratio"])
    out["ols_r2"] = float(model.rsquared)

    return out


def adjusted_association(df: pd.DataFrame) -> dict:
    out: dict = {}
    # Control for key demographics and resources
    covariates = [
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district income
        "feature13",  # % English learners
    ]

    cols = ["stratio"] + covariates
    X = sm.add_constant(df[cols])
    model = sm.OLS(df["testscr"], X, missing="drop").fit()

    out["adj_coef_stratio"] = float(model.params["stratio"])
    out["adj_pvalue_stratio"] = float(model.pvalues["stratio"])
    out["adj_r2"] = float(model.rsquared)

    return out


def trimmed_association(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> dict:
    out: dict = {}
    lo = df["stratio"].quantile(lower_q)
    hi = df["stratio"].quantile(upper_q)
    trimmed = df[(df["stratio"] >= lo) & (df["stratio"] <= hi)].copy()

    out["trim_n"] = int(len(trimmed))
    out["trim_lo"] = float(lo)
    out["trim_hi"] = float(hi)

    corr = trimmed["stratio"].corr(trimmed["testscr"])
    out["trim_corr_stratio_testscr"] = float(corr)

    X = sm.add_constant(trimmed["stratio"])
    model = sm.OLS(trimmed["testscr"], X, missing="drop").fit()
    out["trim_ols_coef_stratio"] = float(model.params["stratio"])
    out["trim_ols_pvalue_stratio"] = float(model.pvalues["stratio"])
    out["trim_ols_r2"] = float(model.rsquared)

    return out


def main() -> None:
    df = load_and_engineer()
    simple = simple_association(df)
    adjusted = adjusted_association(df)
    trimmed = trimmed_association(df)

    summary = {
        "n": int(len(df)),
        "stratio_summary": {
            "mean": float(df["stratio"].mean()),
            "std": float(df["stratio"].std()),
            "min": float(df["stratio"].min()),
            "max": float(df["stratio"].max()),
        },
        "testscr_summary": {
            "mean": float(df["testscr"].mean()),
            "std": float(df["testscr"].std()),
            "min": float(df["testscr"].min()),
            "max": float(df["testscr"].max()),
        },
    }
    summary.update(simple)
    summary.update(adjusted)
    summary.update(trimmed)

    # Print a compact JSON summary so we can interpret results.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
