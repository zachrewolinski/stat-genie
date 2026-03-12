import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map shuffled column names to their semantic meaning using info.json
    # Total enrollment and number of teachers
    df["enrollment"] = df["english"]
    df["n_teachers"] = df["students"]

    # Construct student–teacher ratio (students per teacher)
    df["stratio"] = df["enrollment"] / df["n_teachers"]

    # Academic performance: average of reading and math scores
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["testscr"] = df[["read_score", "math_score"]].mean(axis=1)

    # Key covariates (semantic mapping from info.json)
    df["pct_calworks"] = df["school"]  # Percent qualifying for CalWorks
    df["pct_lunch"] = df["computer"]  # Percent qualifying for reduced-price lunch
    df["pct_ell"] = df["rownames"]  # Percent English learners
    df["expn_stu"] = df["grades"]  # Expenditure per student
    df["avg_income"] = df["income"]  # District average income (thousand USD)

    # Drop rows with any missing values in variables used
    cols = [
        "testscr",
        "stratio",
        "pct_calworks",
        "pct_lunch",
        "pct_ell",
        "expn_stu",
        "avg_income",
    ]
    df_model = df[cols].dropna()

    # Simple correlation between student–teacher ratio and test score
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit()
    coef_simple = model_simple.params["stratio"]
    se_simple = model_simple.bse["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression controlling for key demographics and resources
    formula_full = (
        "testscr ~ stratio + pct_calworks + pct_lunch + pct_ell + expn_stu + avg_income"
    )
    model_full = smf.ols(formula_full, data=df_model).fit()
    coef_full = model_full.params["stratio"]
    se_full = model_full.bse["stratio"]
    p_full = model_full.pvalues["stratio"]
    r2_full = model_full.rsquared

    # Summaries to help interpret results
    summary = {
        "n_obs": int(df_model.shape[0]),
        "stratio": {
            "mean": float(df_model["stratio"].mean()),
            "std": float(df_model["stratio"].std()),
            "min": float(df_model["stratio"].min()),
            "max": float(df_model["stratio"].max()),
        },
        "testscr": {
            "mean": float(df_model["testscr"].mean()),
            "std": float(df_model["testscr"].std()),
            "min": float(df_model["testscr"].min()),
            "max": float(df_model["testscr"].max()),
        },
        "correlation": float(corr),
        "simple_regression": {
            "coef_stratio": float(coef_simple),
            "se_stratio": float(se_simple),
            "p_stratio": float(p_simple),
            "r2": float(r2_simple),
        },
        "full_regression": {
            "coef_stratio": float(coef_full),
            "se_stratio": float(se_full),
            "p_stratio": float(p_full),
            "r2": float(r2_full),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

