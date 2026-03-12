import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables based on info.json descriptions.
    # english: total enrollment, students: number of teachers.
    df["str"] = df["english"] / df["students"]

    # district: average reading score, expenditure: average math score.
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop any rows with missing values in variables used below.
    vars_needed = ["str", "testscr", "income", "school", "computer", "rownames", "grades"]
    df_model = df.dropna(subset=vars_needed)

    # Simple bivariate relationship: testscr ~ str
    X_simple = sm.add_constant(df_model["str"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()

    # Multivariate model with socioeconomic and resource controls.
    predictors = ["str", "income", "school", "computer", "rownames", "grades"]
    X_multi = sm.add_constant(df_model[predictors])
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()

    # Collect key statistics for later reasoning.
    summary = {
        "n_obs": int(df_model.shape[0]),
        "str": {
            "mean": float(df_model["str"].mean()),
            "std": float(df_model["str"].std()),
        },
        "testscr": {
            "mean": float(df_model["testscr"].mean()),
            "std": float(df_model["testscr"].std()),
        },
        "corr_str_testscr": float(df_model["str"].corr(df_model["testscr"])),
        "simple": {
            "coef_str": float(model_simple.params["str"]),
            "pvalue_str": float(model_simple.pvalues["str"]),
            "r2": float(model_simple.rsquared),
        },
        "multi": {
            "coef_str": float(model_multi.params["str"]),
            "pvalue_str": float(model_multi.pvalues["str"]),
            "r2": float(model_multi.rsquared),
        },
    }

    # Write numeric results for inspection.
    Path("analysis_results.json").write_text(json.dumps(summary, indent=2))

    # Also print a concise summary to stdout for interactive review.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

