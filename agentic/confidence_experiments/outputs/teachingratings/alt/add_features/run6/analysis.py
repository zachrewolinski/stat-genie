import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("teachingratings.csv")

    # Keep rows with the core variables for analysis
    df = df.dropna(subset=["beauty", "eval"]).copy()

    n = len(df)
    sd_beauty = df["beauty"].std()
    sd_eval = df["eval"].std()

    corr = df["beauty"].corr(df["eval"])

    # Simple bivariate model
    model1 = smf.ols("eval ~ beauty", data=df).fit()
    model1_rob = model1.get_robustcov_results(cov_type="HC3")

    # Multivariable model with common controls from the dataset
    model2 = smf.ols(
        "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents",
        data=df,
    ).fit()
    model2_rob = model2.get_robustcov_results(cov_type="HC3")

    def coef_p(model):
        names = list(model.model.exog_names)
        beauty_idx = names.index("beauty")
        return float(model.params[beauty_idx]), float(model.pvalues[beauty_idx])

    b1, p1 = coef_p(model1_rob)
    b2, p2 = coef_p(model2_rob)

    # Effect size per 1 SD change in beauty
    effect_sd_1 = b1 * sd_beauty
    effect_sd_2 = b2 * sd_beauty

    results = {
        "n": n,
        "corr_beauty_eval": corr,
        "sd_beauty": sd_beauty,
        "sd_eval": sd_eval,
        "model1_coef": b1,
        "model1_p": p1,
        "model1_effect_per_sd_beauty": effect_sd_1,
        "model2_coef": b2,
        "model2_p": p2,
        "model2_effect_per_sd_beauty": effect_sd_2,
    }

    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
