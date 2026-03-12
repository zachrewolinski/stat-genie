import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "teachingratings.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    # basic sanity
    n = len(df)
    # Pearson correlation between beauty and eval
    corr, corr_p = stats.pearsonr(df["beauty"], df["eval"])

    # Simple OLS
    model_simple = smf.ols("eval ~ beauty", data=df).fit()

    # OLS with controls
    # Convert categorical to C() for statsmodels
    formula_controls = (
        "eval ~ beauty + age + students + allstudents "
        "+ C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)"
    )
    model_controls = smf.ols(formula_controls, data=df).fit()

    # Standardized effect size (beta) for beauty in simple model
    # beta = b * (sd_x / sd_y)
    sd_x = df["beauty"].std(ddof=1)
    sd_y = df["eval"].std(ddof=1)
    beta_simple = model_simple.params["beauty"] * (sd_x / sd_y)

    # Partial effect size: semi-partial R^2 for beauty in controls
    # Compute change in R^2 when removing beauty
    model_controls_no_beauty = smf.ols(
        "eval ~ age + students + allstudents + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)",
        data=df,
    ).fit()
    r2_full = model_controls.rsquared
    r2_reduced = model_controls_no_beauty.rsquared
    semi_partial_r2 = max(r2_full - r2_reduced, 0.0)

    results = {
        "n": n,
        "corr": corr,
        "corr_p": corr_p,
        "simple_coef": model_simple.params["beauty"],
        "simple_p": model_simple.pvalues["beauty"],
        "simple_ci": model_simple.conf_int().loc["beauty"].tolist(),
        "simple_r2": model_simple.rsquared,
        "beta_simple": beta_simple,
        "controls_coef": model_controls.params["beauty"],
        "controls_p": model_controls.pvalues["beauty"],
        "controls_ci": model_controls.conf_int().loc["beauty"].tolist(),
        "controls_r2": r2_full,
        "controls_r2_reduced": r2_reduced,
        "semi_partial_r2": semi_partial_r2,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
