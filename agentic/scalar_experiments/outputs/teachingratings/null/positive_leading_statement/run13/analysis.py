import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "teachingratings.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic sanity checks
    n = len(df)
    missing = df.isna().sum().to_dict()

    # Correlation between beauty and eval
    corr = stats.pearsonr(df["beauty"], df["eval"])

    # Simple OLS
    model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

    # OLS with controls
    formula = (
        "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + "
        "C(division) + C(native) + C(tenure) + students + allstudents"
    )
    model_controls = smf.ols(formula, data=df).fit(cov_type="HC3")

    # Effect size for 1 SD change in beauty
    beauty_sd = df["beauty"].std()
    effect_sd_simple = model_simple.params["beauty"] * beauty_sd
    effect_sd_controls = model_controls.params["beauty"] * beauty_sd

    # R-squared
    r2_simple = model_simple.rsquared
    r2_controls = model_controls.rsquared

    results = {
        "n": n,
        "missing": missing,
        "corr_r": corr.statistic,
        "corr_p": corr.pvalue,
        "simple_coef": model_simple.params["beauty"],
        "simple_se": model_simple.bse["beauty"],
        "simple_p": model_simple.pvalues["beauty"],
        "simple_r2": r2_simple,
        "controls_coef": model_controls.params["beauty"],
        "controls_se": model_controls.bse["beauty"],
        "controls_p": model_controls.pvalues["beauty"],
        "controls_r2": r2_controls,
        "beauty_sd": beauty_sd,
        "effect_sd_simple": effect_sd_simple,
        "effect_sd_controls": effect_sd_controls,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
