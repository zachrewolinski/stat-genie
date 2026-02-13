import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    return df


def simple_correlation(df: pd.DataFrame) -> float:
    return df["stratio"].corr(df["avg_score"])


def run_regressions(df: pd.DataFrame):
    # Bivariate: avg_score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()

    # Multivariate: control for key demographics and resources
    controls = [
        "income",
        "english",
        "calworks",
        "lunch",
        "expenditure",
        "computer",
        "students",
    ]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(df["avg_score"], X_multi).fit()

    return model_simple, model_multi


def main():
    data_path = Path("caschools.csv")
    df = load_data(data_path)

    corr = simple_correlation(df)
    model_simple, model_multi = run_regressions(df)

    stratio_coef_simple = model_simple.params["stratio"]
    stratio_p_simple = model_simple.pvalues["stratio"]

    stratio_coef_multi = model_multi.params["stratio"]
    stratio_p_multi = model_multi.pvalues["stratio"]

    summary = {
        "n": int(df.shape[0]),
        "stratio_mean": float(df["stratio"].mean()),
        "avg_score_mean": float(df["avg_score"].mean()),
        "corr_stratio_avg_score": float(corr),
        "simple": {
            "coef_stratio": float(stratio_coef_simple),
            "p_stratio": float(stratio_p_simple),
            "r2": float(model_simple.rsquared),
        },
        "multi": {
            "coef_stratio": float(stratio_coef_multi),
            "p_stratio": float(stratio_p_multi),
            "r2": float(model_multi.rsquared),
        },
    }

    # Save a machine-readable summary for inspection.
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))

    # Also print a concise text summary to stdout for manual interpretation.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

