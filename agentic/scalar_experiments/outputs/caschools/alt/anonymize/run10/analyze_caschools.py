import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio (students per teacher)
    df["stratio"] = df["feature6"] / df["feature7"]

    # Construct overall test score as the average of reading and math
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop rows with invalid ratios or scores
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "testscr"])

    # Basic correlation between student-teacher ratio and test scores
    corr, corr_p = stats.pearsonr(df["stratio"], df["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    available_controls = [c for c in controls if c in df.columns]
    X_full = sm.add_constant(df[["stratio"] + available_controls])
    model_full = sm.OLS(df["testscr"], X_full).fit()

    results = {
        "n_obs": int(df.shape[0]),
        "stratio_mean": float(df["stratio"].mean()),
        "testscr_mean": float(df["testscr"].mean()),
        "corr_stratio_testscr": float(corr),
        "corr_pvalue": float(corr_p),
        "simple_slope": float(model_simple.params["stratio"]),
        "simple_pvalue": float(model_simple.pvalues["stratio"]),
        "simple_r2": float(model_simple.rsquared),
        "full_slope": float(model_full.params["stratio"]),
        "full_pvalue": float(model_full.pvalues["stratio"]),
        "full_r2": float(model_full.rsquared),
        "controls_used": available_controls,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

