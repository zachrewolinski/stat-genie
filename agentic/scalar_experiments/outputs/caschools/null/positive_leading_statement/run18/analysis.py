import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used in models
    base_vars = ["stratio", "avg_score"]
    control_vars = [
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
        "students",
    ]
    df_model = df[base_vars + control_vars].dropna()

    # Simple bivariate regression: avg_score ~ stratio
    X1 = sm.add_constant(df_model["stratio"])
    y = df_model["avg_score"]
    model1 = sm.OLS(y, X1).fit()

    # Multiple regression with controls
    X2 = sm.add_constant(df_model[["stratio"] + control_vars])
    model2 = sm.OLS(y, X2).fit()

    # Collect key statistics
    results = {
        "n_obs": int(df_model.shape[0]),
        "stratio": {
            "mean": float(df_model["stratio"].mean()),
            "std": float(df_model["stratio"].std()),
        },
        "avg_score": {
            "mean": float(df_model["avg_score"].mean()),
            "std": float(df_model["avg_score"].std()),
        },
        "bivariate": {
            "coef_stratio": float(model1.params["stratio"]),
            "pval_stratio": float(model1.pvalues["stratio"]),
            "r_squared": float(model1.rsquared),
        },
        "multivariate": {
            "coef_stratio": float(model2.params["stratio"]),
            "pval_stratio": float(model2.pvalues["stratio"]),
            "r_squared": float(model2.rsquared),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

