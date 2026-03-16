import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def summarize_model(model):
    return {
        "coef": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "r2": getattr(model, "rsquared", None),
        "n": int(model.nobs),
    }


def main():
    df = pd.read_csv("hurricane.csv")

    # Variable mapping from info.json
    femininity = df["feature4"]  # masculinity-femininity index
    female_indicator = df["feature6"]  # 0 male, 1 female
    deaths = df["feature8"]
    damage_2013 = df["feature9"]
    damage_2015 = df["feature14"]
    category = df["feature7"]
    min_pressure = df["feature5"]
    max_wind = df["feature13"]
    year = df["feature2"]

    df = df.copy()
    df["log_deaths"] = np.log1p(deaths)
    df["log_damage_2013"] = np.log1p(damage_2013)
    df["log_damage_2015"] = np.log1p(damage_2015)

    # Bivariate correlations
    corrs = {
        "fem_index_log_deaths_pearson": df[["feature4", "log_deaths"]].corr().iloc[0, 1],
        "fem_index_log_deaths_spearman": df[["feature4", "log_deaths"]].corr(method="spearman").iloc[0, 1],
        "female_indicator_log_deaths_pearson": df[["feature6", "log_deaths"]].corr().iloc[0, 1],
        "female_indicator_log_deaths_spearman": df[["feature6", "log_deaths"]].corr(method="spearman").iloc[0, 1],
    }

    # Regression models
    # Control for storm intensity (category, min pressure, max wind) and year
    model1 = smf.ols("log_deaths ~ feature4 + feature7 + feature5 + feature13 + feature2", data=df).fit(cov_type="HC3")
    model2 = smf.ols("log_deaths ~ feature6 + feature7 + feature5 + feature13 + feature2", data=df).fit(cov_type="HC3")

    model3 = smf.ols("log_damage_2013 ~ feature4 + feature7 + feature5 + feature13 + feature2", data=df).fit(cov_type="HC3")
    model4 = smf.ols("log_damage_2013 ~ feature6 + feature7 + feature5 + feature13 + feature2", data=df).fit(cov_type="HC3")

    model5 = smf.ols("log_damage_2015 ~ feature4 + feature7 + feature5 + feature13 + feature2", data=df).fit(cov_type="HC3")
    model6 = smf.ols("log_damage_2015 ~ feature6 + feature7 + feature5 + feature13 + feature2", data=df).fit(cov_type="HC3")

    out = {
        "n": int(len(df)),
        "corrs": corrs,
        "model_fem_log_deaths": summarize_model(model1),
        "model_female_log_deaths": summarize_model(model2),
        "model_fem_log_damage_2013": summarize_model(model3),
        "model_female_log_damage_2013": summarize_model(model4),
        "model_fem_log_damage_2015": summarize_model(model5),
        "model_female_log_damage_2015": summarize_model(model6),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
