import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Reconstruct semantic variables based on info.json descriptions.
    enrollment = df["english"]  # total enrollment
    teachers = df["students"]  # number of teachers

    # Student-teacher ratio: students per teacher.
    stratio = enrollment / teachers

    read_score = df["district"]  # average reading score
    math_score = df["expenditure"]  # average math score
    testscr = (read_score + math_score) / 2.0

    # Socioeconomic and demographic controls.
    income = df["income"]
    pct_calworks = df["school"]  # percent qualifying for CalWorks
    pct_lunch = df["computer"]  # percent qualifying for reduced-price lunch
    pct_ell = df["rownames"]  # percent English learners
    expn_stu = df["grades"]  # expenditure per student

    base_df = pd.DataFrame(
        {
            "testscr": testscr,
            "stratio": stratio,
            "income": income,
            "pct_calworks": pct_calworks,
            "pct_lunch": pct_lunch,
            "pct_ell": pct_ell,
            "expn_stu": expn_stu,
        }
    ).dropna()

    # Restrict to a more plausible range of student-teacher ratios
    # to check robustness against extreme outliers.
    trimmed_df = base_df[(base_df["stratio"] >= 5) & (base_df["stratio"] <= 40)].copy()

    # Basic descriptive statistics.
    corr = base_df["testscr"].corr(base_df["stratio"])
    corr_trim = trimmed_df["testscr"].corr(trimmed_df["stratio"])

    # Simple bivariate regression: testscr on student-teacher ratio.
    model_simple = smf.ols("testscr ~ stratio", data=base_df).fit()
    model_simple_trim = smf.ols("testscr ~ stratio", data=trimmed_df).fit()

    # Multiple regression controlling for key covariates.
    model_controls = smf.ols(
        "testscr ~ stratio + income + pct_calworks + pct_lunch + pct_ell + expn_stu",
        data=base_df,
    ).fit()
    model_controls_trim = smf.ols(
        "testscr ~ stratio + income + pct_calworks + pct_lunch + pct_ell + expn_stu",
        data=trimmed_df,
    ).fit()

    # Summarize key quantities needed for the narrative explanation.
    results = {
        "n_obs": int(base_df.shape[0]),
        "n_obs_trim": int(trimmed_df.shape[0]),
        "corr_testscr_stratio": float(corr),
        "corr_testscr_stratio_trim": float(corr_trim),
        "simple_coef_stratio": float(model_simple.params["stratio"]),
        "simple_pvalue_stratio": float(model_simple.pvalues["stratio"]),
        "simple_r2": float(model_simple.rsquared),
        "simple_coef_stratio_trim": float(model_simple_trim.params["stratio"]),
        "simple_pvalue_stratio_trim": float(model_simple_trim.pvalues["stratio"]),
        "simple_r2_trim": float(model_simple_trim.rsquared),
        "controls_coef_stratio": float(model_controls.params["stratio"]),
        "controls_pvalue_stratio": float(model_controls.pvalues["stratio"]),
        "controls_r2": float(model_controls.rsquared),
        "controls_coef_stratio_trim": float(model_controls_trim.params["stratio"]),
        "controls_pvalue_stratio_trim": float(
            model_controls_trim.pvalues["stratio"]
        ),
        "controls_r2_trim": float(model_controls_trim.rsquared),
        "stratio_mean": float(base_df["stratio"].mean()),
        "stratio_std": float(base_df["stratio"].std(ddof=1)),
        "testscr_mean": float(base_df["testscr"].mean()),
        "testscr_std": float(base_df["testscr"].std(ddof=1)),
        "stratio_mean_trim": float(trimmed_df["stratio"].mean()),
        "stratio_std_trim": float(trimmed_df["stratio"].std(ddof=1)),
        "testscr_mean_trim": float(trimmed_df["testscr"].mean()),
        "testscr_std_trim": float(trimmed_df["testscr"].std(ddof=1)),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
