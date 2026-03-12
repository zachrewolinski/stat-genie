import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data).fit(disp=False)
    return model


def lr_test(model_restricted, model_full):
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    pval = stats.chi2.sf(lr_stat, df_diff)
    return float(lr_stat), int(df_diff), float(pval)


def main():
    df = pd.read_csv("boxes.csv")

    # Encode key outcomes
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)
    social_rate = df["social_choice"].mean()

    df_social = df[df["social_choice"] == 1].copy()
    df_social["majority_choice"] = (df_social["majority_first"] == 2).astype(int)
    majority_rate = df_social["majority_choice"].mean()

    # Logistic models for reliance on social information
    m_sc_age = fit_logit("social_choice ~ age", df)
    m_sc_site = fit_logit("social_choice ~ C(y)", df)
    m_sc_full = fit_logit("social_choice ~ age + C(y)", df)

    sc_age_lr, sc_age_df, sc_age_p = lr_test(m_sc_site, m_sc_full)
    sc_site_lr, sc_site_df, sc_site_p = lr_test(m_sc_age, m_sc_full)

    # Logistic models for preference for majority cues (conditional on social choice)
    m_mc_age = fit_logit("majority_choice ~ age", df_social)
    m_mc_site = fit_logit("majority_choice ~ C(y)", df_social)
    m_mc_full = fit_logit("majority_choice ~ age + C(y)", df_social)

    mc_age_lr, mc_age_df, mc_age_p = lr_test(m_mc_site, m_mc_full)
    mc_site_lr, mc_site_df, mc_site_p = lr_test(m_mc_age, m_mc_full)

    # Simple descriptive summaries by age (quartiles) and site
    df["age_group"] = pd.qcut(df["age"], q=4, duplicates="drop")
    df["age_group_label"] = df["age_group"].astype(str)
    social_by_age = (
        df.groupby("age_group_label")["social_choice"]
        .mean()
        .reset_index()
        .rename(columns={"age_group_label": "age_group"})
        .to_dict("records")
    )
    social_by_site = (
        df.groupby("y")["social_choice"].mean().reset_index().to_dict("records")
    )

    df_social["age_group"] = pd.qcut(df_social["age"], q=4, duplicates="drop")
    df_social["age_group_label"] = df_social["age_group"].astype(str)
    majority_by_age = (
        df_social.groupby("age_group_label")["majority_choice"]
        .mean()
        .reset_index()
        .rename(columns={"age_group_label": "age_group"})
        .to_dict("records")
    )
    majority_by_site = (
        df_social.groupby("y")["majority_choice"].mean().reset_index().to_dict("records")
    )

    # Collect key statistics into a JSON file for inspection
    results = {
        "overall": {
            "n_total": int(len(df)),
            "social_rate": float(social_rate),
            "majority_rate_given_social": float(majority_rate),
        },
        "social_choice_models": {
            "age_effect_lr": sc_age_lr,
            "age_effect_df": sc_age_df,
            "age_effect_p": sc_age_p,
            "site_effect_lr": sc_site_lr,
            "site_effect_df": sc_site_df,
            "site_effect_p": sc_site_p,
            "age_coef": float(m_sc_full.params["age"]),
        },
        "majority_choice_models": {
            "age_effect_lr": mc_age_lr,
            "age_effect_df": mc_age_df,
            "age_effect_p": mc_age_p,
            "site_effect_lr": mc_site_lr,
            "site_effect_df": mc_site_df,
            "site_effect_p": mc_site_p,
            "age_coef": float(m_mc_full.params["age"]),
        },
        "descriptives": {
            "social_by_age_group": social_by_age,
            "social_by_site": social_by_site,
            "majority_by_age_group": majority_by_age,
            "majority_by_site": majority_by_site,
        },
    }

    out_path = Path("results.json")
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
