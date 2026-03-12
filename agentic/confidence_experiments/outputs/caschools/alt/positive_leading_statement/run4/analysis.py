import json

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: students per teacher (lower is better).
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2

    # Drop any rows with missing values in the variables of interest.
    df = df.dropna(subset=["stratio", "testscr", "income", "english", "calworks", "lunch"])

    # Simple bivariate relationship.
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()

    # Relationship controlling for key demographics and SES variables.
    model_controls = smf.ols(
        "testscr ~ stratio + income + english + calworks + lunch",
        data=df,
    ).fit()

    corr = df["stratio"].corr(df["testscr"])

    results = {
        "n_obs": int(df.shape[0]),
        "corr_stratio_testscr": float(corr),
        "simple_coef_stratio": float(model_simple.params["stratio"]),
        "simple_pvalue_stratio": float(model_simple.pvalues["stratio"]),
        "simple_r2": float(model_simple.rsquared),
        "controls_coef_stratio": float(model_controls.params["stratio"]),
        "controls_pvalue_stratio": float(model_controls.pvalues["stratio"]),
        "controls_r2": float(model_controls.rsquared),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

