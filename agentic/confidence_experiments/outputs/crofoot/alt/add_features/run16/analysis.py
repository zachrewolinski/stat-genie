import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main():
    df = pd.read_csv("crofoot.csv")

    # Key variables
    df = df.copy()
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["rel_size_ratio"] = df["n_focal"] / df["n_other"]
    # Positive means contest closer to focal group's home range center
    df["rel_location"] = df["dist_other"] - df["dist_focal"]

    # Drop rows with missing key variables
    key_cols = ["win", "rel_size", "rel_size_ratio", "rel_location"]
    df_model = df.dropna(subset=key_cols)

    # Standardize predictors for effect size comparability
    df_model["rel_size_z"] = (df_model["rel_size"] - df_model["rel_size"].mean()) / df_model["rel_size"].std(ddof=0)
    df_model["rel_location_z"] = (df_model["rel_location"] - df_model["rel_location"].mean()) / df_model["rel_location"].std(ddof=0)

    # Logistic regression with standardized predictors
    X = sm.add_constant(df_model[["rel_size_z", "rel_location_z"]])
    y = df_model["win"]
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit()

    # Alternative model using ratio (also standardized)
    df_model["rel_size_ratio_z"] = (df_model["rel_size_ratio"] - df_model["rel_size_ratio"].mean()) / df_model["rel_size_ratio"].std(ddof=0)
    X_ratio = sm.add_constant(df_model[["rel_size_ratio_z", "rel_location_z"]])
    model_ratio = sm.GLM(y, X_ratio, family=sm.families.Binomial())
    res_ratio = model_ratio.fit()

    # Simple point-biserial correlations
    corr_size = stats.pointbiserialr(df_model["win"], df_model["rel_size"])
    corr_loc = stats.pointbiserialr(df_model["win"], df_model["rel_location"])

    # Odds ratios per 1 SD change
    or_size = float(np.exp(res.params["rel_size_z"]))
    or_loc = float(np.exp(res.params["rel_location_z"]))

    summary = {
        "n": int(df_model.shape[0]),
        "logit_coef": res.params.to_dict(),
        "logit_pvalues": res.pvalues.to_dict(),
        "logit_or_per_sd": {"rel_size_z": or_size, "rel_location_z": or_loc},
        "ratio_model_pvalues": res_ratio.pvalues.to_dict(),
        "corr_rel_size": {"r": float(corr_size.correlation), "p": float(corr_size.pvalue)},
        "corr_rel_location": {"r": float(corr_loc.correlation), "p": float(corr_loc.pvalue)},
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
