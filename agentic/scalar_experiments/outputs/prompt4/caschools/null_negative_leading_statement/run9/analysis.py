import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load metadata (not strictly required for computation, but documents context)
    info_path = Path("info.json")
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
    else:
        info = {}

    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = 0.5 * (df["read"] + df["math"])

    # Drop any missing values in the variables used
    sub = df[["testscr", "stratio", "income", "english", "lunch"]].dropna()

    # Simple correlation
    r, p_corr = stats.pearsonr(sub["stratio"], sub["testscr"])

    # Simple bivariate OLS: testscr ~ stratio
    X1 = sm.add_constant(sub["stratio"])
    model1 = sm.OLS(sub["testscr"], X1).fit()

    # Multiple regression controlling for key demographics
    X2 = sm.add_constant(sub[["stratio", "income", "english", "lunch"]])
    model2 = sm.OLS(sub["testscr"], X2).fit()

    results = {
        "research_question": info.get("research_questions", [None])[0],
        "n_obs": int(len(sub)),
        "stratio_summary": {
            "mean": float(sub["stratio"].mean()),
            "std": float(sub["stratio"].std()),
            "min": float(sub["stratio"].min()),
            "max": float(sub["stratio"].max()),
        },
        "testscr_summary": {
            "mean": float(sub["testscr"].mean()),
            "std": float(sub["testscr"].std()),
            "min": float(sub["testscr"].min()),
            "max": float(sub["testscr"].max()),
        },
        "correlation": {
            "r": float(r),
            "p_value": float(p_corr),
        },
        "bivariate_regression": {
            "coef_stratio": float(model1.params["stratio"]),
            "p_value_stratio": float(model1.pvalues["stratio"]),
            "r_squared": float(model1.rsquared),
        },
        "multivariate_regression": {
            "coef_stratio": float(model2.params["stratio"]),
            "p_value_stratio": float(model2.pvalues["stratio"]),
            "r_squared": float(model2.rsquared),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

