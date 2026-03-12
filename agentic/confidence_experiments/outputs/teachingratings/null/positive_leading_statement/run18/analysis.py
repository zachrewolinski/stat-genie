import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("teachingratings.csv")

    # Basic stats
    n = len(df)
    beauty_sd = df["beauty"].std()
    eval_sd = df["eval"].std()

    # Correlation
    r, p = stats.pearsonr(df["beauty"], df["eval"])

    # OLS models
    model1 = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC1")

    formula = (
        "eval ~ beauty + age + C(gender) + C(minority) + C(native) + "
        "C(tenure) + C(division) + C(credits) + students"
    )
    model2 = smf.ols(formula, data=df).fit(cov_type="HC1")
    model2_cluster = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["prof"]}
    )

    out = {
        "n": n,
        "beauty_sd": beauty_sd,
        "eval_sd": eval_sd,
        "pearson_r": r,
        "pearson_p": p,
        "model1": {
            "coef_beauty": model1.params["beauty"],
            "se_beauty": model1.bse["beauty"],
            "p_beauty": model1.pvalues["beauty"],
            "r2": model1.rsquared,
            "effect_1sd": model1.params["beauty"] * beauty_sd,
        },
        "model2": {
            "coef_beauty": model2.params["beauty"],
            "se_beauty": model2.bse["beauty"],
            "p_beauty": model2.pvalues["beauty"],
            "r2": model2.rsquared,
            "effect_1sd": model2.params["beauty"] * beauty_sd,
        },
        "model2_cluster": {
            "coef_beauty": model2_cluster.params["beauty"],
            "se_beauty": model2_cluster.bse["beauty"],
            "p_beauty": model2_cluster.pvalues["beauty"],
            "r2": model2_cluster.rsquared,
            "effect_1sd": model2_cluster.params["beauty"] * beauty_sd,
        },
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
