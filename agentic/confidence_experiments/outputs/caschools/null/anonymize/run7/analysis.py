import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    info_path = Path("info.json")

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["feature6"] / df["feature7"]  # students per teacher
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0  # average of reading and math

    # Simple bivariate association
    y = df["testscr"]
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()

    # Multivariable association controlling for observed district characteristics
    controls = df[["feature12", "feature11", "feature8", "feature9", "feature13"]].copy()
    controls = sm.add_constant(controls)
    X_multi = controls.join(df["stratio"])
    model_multi = sm.OLS(y, X_multi).fit()

    # Correlations
    corr_testscr_str = df["testscr"].corr(df["stratio"])
    corr_read_str = df["feature14"].corr(df["stratio"])
    corr_math_str = df["feature15"].corr(df["stratio"])

    # Collect key statistics for inspection
    stats = {
        "research_question": info["research_questions"][0],
        "n_obs": int(df.shape[0]),
        "stratio_mean": float(df["stratio"].mean()),
        "stratio_std": float(df["stratio"].std()),
        "testscr_mean": float(df["testscr"].mean()),
        "testscr_std": float(df["testscr"].std()),
        "corr_testscr_str": float(corr_testscr_str),
        "corr_read_str": float(corr_read_str),
        "corr_math_str": float(corr_math_str),
        "simple_coef_str": float(model_simple.params["stratio"]),
        "simple_pvalue_str": float(model_simple.pvalues["stratio"]),
        "multi_coef_str": float(model_multi.params["stratio"]),
        "multi_pvalue_str": float(model_multi.pvalues["stratio"]),
        "simple_r2": float(model_simple.rsquared),
        "multi_r2": float(model_multi.rsquared),
    }

    out_path = Path("analysis_results.json")
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)

    # Also print a concise summary to stdout for quick inspection.
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

