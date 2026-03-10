import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def ols_robust(df, y, x_cols):
    X = sm.add_constant(df[x_cols])
    model = sm.OLS(df[y], X, missing="drop").fit(cov_type="HC3")
    return model


def run():
    df = pd.read_csv("hurricane.csv")

    # outcomes
    df["log_deaths"] = np.log1p(df["alldeaths"])

    # core predictors
    predictors = ["masfem", "category", "wind", "min"]
    predictors_mturk = ["masfem_mturk", "category", "wind", "min"]
    predictors_gender = ["gender_mf", "category", "wind", "min"]

    # drop rows with missing key values
    df_model = df.dropna(subset=predictors + ["log_deaths"]).copy()

    # correlations (Spearman, robust to skew)
    spearman_masfem = stats.spearmanr(df_model["masfem"], df_model["alldeaths"])
    spearman_masfem_log = stats.spearmanr(df_model["masfem"], df_model["log_deaths"])

    # OLS with robust SE
    m1 = ols_robust(df_model, "log_deaths", predictors)

    # alternate femininity measure
    df_mturk = df.dropna(subset=predictors_mturk + ["log_deaths"]).copy()
    m2 = ols_robust(df_mturk, "log_deaths", predictors_mturk)

    # binary gender
    df_gender = df.dropna(subset=predictors_gender + ["log_deaths"]).copy()
    m3 = ols_robust(df_gender, "log_deaths", predictors_gender)

    # print summary stats needed for interpretation
    print("N total:", len(df))
    print("N model masfem:", len(df_model))
    print("Spearman masfem vs deaths: rho=%.3f p=%.3f" % (spearman_masfem.correlation, spearman_masfem.pvalue))
    print("Spearman masfem vs log_deaths: rho=%.3f p=%.3f" % (spearman_masfem_log.correlation, spearman_masfem_log.pvalue))

    def coef_info(model, var):
        return {
            "coef": model.params[var],
            "se": model.bse[var],
            "p": model.pvalues[var],
            "ci_low": model.conf_int().loc[var, 0],
            "ci_high": model.conf_int().loc[var, 1],
        }

    print("OLS log_deaths ~ masfem + category + wind + min (HC3)")
    print(coef_info(m1, "masfem"))

    print("OLS log_deaths ~ masfem_mturk + category + wind + min (HC3)")
    print(coef_info(m2, "masfem_mturk"))

    print("OLS log_deaths ~ gender_mf + category + wind + min (HC3)")
    print(coef_info(m3, "gender_mf"))


if __name__ == "__main__":
    run()
