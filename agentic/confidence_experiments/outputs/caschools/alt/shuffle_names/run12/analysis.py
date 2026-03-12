import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map to semantic variables using info.json descriptions.
    # student-teacher ratio = total enrollment / number of teachers
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop any rows with missing values in variables of interest
    df_model = df[["stratio", "testscr", "income", "school", "computer", "rownames"]].dropna()

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()

    # Multiple regression with key controls
    X_controls = df_model[["stratio", "income", "school", "computer", "rownames"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit()

    # Extract statistics for student-teacher ratio
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    coef_controls = model_controls.params["stratio"]
    pval_controls = model_controls.pvalues["stratio"]
    r2_controls = model_controls.rsquared

    corr = np.corrcoef(df_model["stratio"], df_model["testscr"])[0, 1]

    results = {
        "n_obs": int(df_model.shape[0]),
        "corr_stratio_testscr": float(corr),
        "simple": {
            "coef": float(coef_simple),
            "p_value": float(pval_simple),
            "r_squared": float(r2_simple),
        },
        "controls": {
            "coef": float(coef_controls),
            "p_value": float(pval_controls),
            "r_squared": float(r2_controls),
        },
    }

    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

