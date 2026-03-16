import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Recode outcome variables
    df["use_social"] = df["majority_first"].isin([2, 3]).astype(int)
    df["choose_majority"] = (df["majority_first"] == 2).astype(int)

    # Center age for stability
    df["age_c"] = df["age"] - df["age"].mean()

    # Model 1: reliance on any social information (majority or minority) vs undemonstrated option
    model_social = smf.glm(
        formula="use_social ~ age_c + C(y)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_social_nosite = smf.glm(
        formula="use_social ~ age_c",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Restrict to children who used social information to model majority vs minority choice
    df_social = df[df["use_social"] == 1].copy()
    model_majority = smf.glm(
        formula="choose_majority ~ age_c + C(y)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    model_majority_nosite = smf.glm(
        formula="choose_majority ~ age_c",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    # Extract key statistics
    age_social_p = float(model_social.pvalues.get("age_c", float("nan")))
    age_majority_p = float(model_majority.pvalues.get("age_c", float("nan")))

    # Likelihood-ratio tests for cultural (site) effects
    lr_social_stat = 2 * (model_social.llf - model_social_nosite.llf)
    lr_social_df = int(model_social.df_model - model_social_nosite.df_model)
    lr_social_p = float(1.0 - chi2.cdf(lr_social_stat, lr_social_df))

    lr_majority_stat = 2 * (model_majority.llf - model_majority_nosite.llf)
    lr_majority_df = int(model_majority.df_model - model_majority_nosite.df_model)
    lr_majority_p = float(1.0 - chi2.cdf(lr_majority_stat, lr_majority_df))

    # Overall site effects: range of site coefficients as a simple variation summary
    site_terms_social = [t for t in model_social.params.index if t.startswith("C(y)[T.")]
    site_terms_majority = [t for t in model_majority.params.index if t.startswith("C(y)[T.")]

    site_range_social = float(
        (model_social.params[site_terms_social].max() - model_social.params[site_terms_social].min())
        if site_terms_social
        else 0.0
    )
    site_range_majority = float(
        (model_majority.params[site_terms_majority].max() - model_majority.params[site_terms_majority].min())
        if site_terms_majority
        else 0.0
    )

    summary = {
        "n_total": int(len(df)),
        "n_social": int(df["use_social"].sum()),
        "prop_social": float(df["use_social"].mean()),
        "n_majority_among_social": int(df_social["choose_majority"].sum()),
        "prop_majority_among_social": float(df_social["choose_majority"].mean()),
        "age_social_coef": float(model_social.params.get("age_c", float("nan"))),
        "age_social_p": age_social_p,
        "age_majority_coef": float(model_majority.params.get("age_c", float("nan"))),
        "age_majority_p": age_majority_p,
        "lr_social_stat": float(lr_social_stat),
        "lr_social_df": int(lr_social_df),
        "lr_social_p": float(lr_social_p),
        "lr_majority_stat": float(lr_majority_stat),
        "lr_majority_df": int(lr_majority_df),
        "lr_majority_p": float(lr_majority_p),
        "site_range_social": site_range_social,
        "site_range_majority": site_range_majority,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
