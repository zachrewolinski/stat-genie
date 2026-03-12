import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def fit_logit(y, X, drop_cols=None):
    if drop_cols is not None:
        X = X.drop(columns=drop_cols)
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X).fit(disp=False)
    return model


def likelihood_ratio_test(full_model, reduced_model, df_diff):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Basic derived variables
    df["age"] = df["feature3"].astype(float)
    df["site"] = df["feature5"].astype(int)
    df["gender"] = df["feature2"].astype(int)
    df["majority_first"] = df["feature4"].astype(int)

    # Reliance on social information: chose majority or minority option vs undemonstrated option
    df["used_social_info"] = (df["feature1"].isin([2, 3])).astype(int)

    # Majority preference among children who used social information
    df_si = df[df["used_social_info"] == 1].copy()
    df_si["chose_majority"] = (df_si["feature1"] == 2).astype(int)

    # Design matrices
    # Encode site and gender as categorical via dummies; keep age and majority_first as-is.
    X_full_social = pd.get_dummies(
        df[["age", "site", "gender", "majority_first"]],
        columns=["site", "gender"],
        drop_first=True,
    )
    y_social = df["used_social_info"]

    X_full_majority = pd.get_dummies(
        df_si[["age", "site", "gender", "majority_first"]],
        columns=["site", "gender"],
        drop_first=True,
    )
    y_majority = df_si["chose_majority"]

    # Full models
    model_social_full = fit_logit(y_social, X_full_social)
    model_majority_full = fit_logit(y_majority, X_full_majority)

    # Reduced models for likelihood-ratio tests
    # Remove all site dummies to test cultural variation
    site_cols_social = [c for c in X_full_social.columns if c.startswith("site_")]
    site_cols_majority = [c for c in X_full_majority.columns if c.startswith("site_")]

    model_social_nosite = fit_logit(y_social, X_full_social, drop_cols=site_cols_social)
    model_majority_nosite = fit_logit(
        y_majority, X_full_majority, drop_cols=site_cols_majority
    )

    lr_social_site, p_social_site = likelihood_ratio_test(
        model_social_full,
        model_social_nosite,
        df_diff=len(site_cols_social),
    )
    lr_majority_site, p_majority_site = likelihood_ratio_test(
        model_majority_full,
        model_majority_nosite,
        df_diff=len(site_cols_majority),
    )

    # Reduced models without age to test developmental change
    age_cols_social = [c for c in X_full_social.columns if c == "age"]
    age_cols_majority = [c for c in X_full_majority.columns if c == "age"]

    model_social_noage = fit_logit(y_social, X_full_social, drop_cols=age_cols_social)
    model_majority_noage = fit_logit(
        y_majority, X_full_majority, drop_cols=age_cols_majority
    )

    lr_social_age, p_social_age = likelihood_ratio_test(
        model_social_full,
        model_social_noage,
        df_diff=len(age_cols_social),
    )
    lr_majority_age, p_majority_age = likelihood_ratio_test(
        model_majority_full,
        model_majority_noage,
        df_diff=len(age_cols_majority),
    )

    # Compute some descriptive statistics for effect sizes
    summary = {}

    # Overall probabilities
    social_rate = df["used_social_info"].mean()
    majority_rate = df_si["chose_majority"].mean()

    summary["overall_social_rate"] = float(social_rate)
    summary["overall_majority_rate"] = float(majority_rate)

    # Age effects: predicted probabilities at younger vs older ages
    def predict_by_age(model, X_full, age_value):
        # Start from the median covariate profile in the original design matrix
        row = X_full.median(numeric_only=True).to_frame().T
        row["age"] = age_value
        row = sm.add_constant(row, has_constant="add")
        # Align columns with the model's exogenous design matrix
        row = row.reindex(columns=model.model.exog_names)
        prob = float(model.predict(row)[0])
        return prob

    # Use median site/gender/majority_first configuration
    min_age = df["age"].min()
    max_age = df["age"].max()

    social_prob_young = predict_by_age(
        model_social_full, X_full_social, age_value=min_age
    )
    social_prob_old = predict_by_age(
        model_social_full, X_full_social, age_value=max_age
    )

    majority_prob_young = predict_by_age(
        model_majority_full, X_full_majority, age_value=min_age
    )
    majority_prob_old = predict_by_age(
        model_majority_full, X_full_majority, age_value=max_age
    )

    summary["social_prob_young"] = float(social_prob_young)
    summary["social_prob_old"] = float(social_prob_old)
    summary["majority_prob_young"] = float(majority_prob_young)
    summary["majority_prob_old"] = float(majority_prob_old)

    # Basic spread across sites: mean per-site
    site_social_rates = df.groupby("site")["used_social_info"].mean().to_dict()
    site_majority_rates = df_si.groupby("site")["chose_majority"].mean().to_dict()

    summary["site_social_rates"] = {int(k): float(v) for k, v in site_social_rates.items()}
    summary["site_majority_rates"] = {
        int(k): float(v) for k, v in site_majority_rates.items()
    }

    # Print a concise JSON with key statistics and p-values to stdout
    results = {
        "n_total": int(len(df)),
        "n_social_info": int(df["used_social_info"].sum()),
        "n_majority_sample": int(len(df_si)),
        "p_values": {
            "social_site": float(p_social_site),
            "social_age": float(p_social_age),
            "majority_site": float(p_majority_site),
            "majority_age": float(p_majority_age),
        },
        "lr_stats": {
            "social_site": float(lr_social_site),
            "social_age": float(lr_social_age),
            "majority_site": float(lr_majority_site),
            "majority_age": float(lr_majority_age),
        },
        "summary": summary,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
