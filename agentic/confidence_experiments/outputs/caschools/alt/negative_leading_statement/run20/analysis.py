import json

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: number of students per full-time-equivalent teacher.
    df["str"] = df["students"] / df["teachers"]

    # Overall academic performance as the average of reading and math scores.
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(
        subset=[
            "str",
            "testscr",
            "income",
            "english",
            "lunch",
            "calworks",
            "expenditure",
        ]
    )

    # Correlation between student-teacher ratio and test scores.
    r, p_corr = stats.pearsonr(df["str"], df["testscr"])

    # Simple linear regression: test score on student-teacher ratio.
    model_simple = smf.ols("testscr ~ str", data=df).fit()

    # Regression with key socioeconomic and funding controls.
    model_controls = smf.ols(
        "testscr ~ str + income + english + lunch + calworks + expenditure",
        data=df,
    ).fit()

    results = {
        "n": int(df.shape[0]),
        "str_mean": float(df["str"].mean()),
        "testscr_mean": float(df["testscr"].mean()),
        "corr_str_testscr": float(r),
        "corr_p_value": float(p_corr),
        "simple_coef_str": float(model_simple.params["str"]),
        "simple_coef_p_value": float(model_simple.pvalues["str"]),
        "simple_r2": float(model_simple.rsquared),
        "controls_coef_str": float(model_controls.params["str"]),
        "controls_coef_p_value": float(model_controls.pvalues["str"]),
        "controls_r2": float(model_controls.rsquared),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

