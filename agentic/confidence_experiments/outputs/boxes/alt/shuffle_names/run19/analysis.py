import json

import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def fit_logit(formula, data, label):
    try:
        model = smf.logit(formula, data=data).fit(disp=False)
        print(f"\n=== Logistic regression for {label} ===")
        print(model.summary())
        return model
    except Exception as exc:  # pragma: no cover - defensive
        print(f"\n[WARN] Logistic regression for {label} failed: {exc}")
        return None


def lr_test(full_model, reduced_model, label):
    if full_model is None or reduced_model is None:
        return None
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    print(
        f"\nLikelihood ratio test for {label}: "
        f"LR={lr_stat:.3f}, df={df_diff:.0f}, p={p_value:.4g}"
    )
    return p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Outcome coding: 1 = undemonstrated option, 2 = majority, 3 = minority
    df["social_choice"] = (df["majority_first"] != 1).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    print("Dataset shape:", df.shape)
    print("\nOutcome counts (1=undemonstrated, 2=majority, 3=minority):")
    print(df["majority_first"].value_counts().sort_index())

    # Descriptive: social vs asocial by site
    print("\nProportion of social choices by site (y):")
    print(
        df.groupby("y")["social_choice"].mean().rename("prop_social")
    )

    # Descriptive: majority preference among social choices by site
    df_social = df[df["social_choice"] == 1].copy()
    print("\nProportion of majority choices among social choices by site (y):")
    print(
        df_social.groupby("y")["majority_choice"].mean().rename("prop_majority_given_social")
    )

    # Logistic models for social vs asocial
    social_full = fit_logit("social_choice ~ age + C(y)", df, "social_choice ~ age + C(y)")
    social_age_only = fit_logit("social_choice ~ age", df, "social_choice ~ age")

    # Logistic models for majority vs other among social choices
    majority_full = fit_logit(
        "majority_choice ~ age + C(y)", df_social, "majority_choice ~ age + C(y)"
    )
    majority_age_only = fit_logit(
        "majority_choice ~ age", df_social, "majority_choice ~ age"
    )

    # Extract p-values for age effects
    age_p_social = None
    age_p_majority = None
    if social_full is not None and "age" in social_full.pvalues:
        age_p_social = social_full.pvalues["age"]
        print(f"\nAge effect (social vs asocial): coef={social_full.params['age']:.3f}, "
              f"p={age_p_social:.4g}")
    if majority_full is not None and "age" in majority_full.pvalues:
        age_p_majority = majority_full.pvalues["age"]
        print(
            f"Age effect (majority vs other among social): "
            f"coef={majority_full.params['age']:.3f}, p={age_p_majority:.4g}"
        )

    # Likelihood-ratio tests for site (culture) effects
    site_p_social = lr_test(
        social_full, social_age_only, "site effect on social vs asocial"
    )
    site_p_majority = lr_test(
        majority_full, majority_age_only, "site effect on majority vs other (social only)"
    )

    # Chi-square tests as a robustness check
    print("\nChi-square test: social vs asocial by site (y):")
    social_table = pd.crosstab(df["y"], df["social_choice"])
    chi2_social, p_social_chi, _, _ = stats.chi2_contingency(social_table)
    print(f"chi2={chi2_social:.3f}, df={social_table.size - 1}, p={p_social_chi:.4g}")

    print("\nChi-square test: majority vs minority/other among social choices by site (y):")
    majority_table = pd.crosstab(df_social["y"], df_social["majority_choice"])
    chi2_majority, p_majority_chi, _, _ = stats.chi2_contingency(majority_table)
    print(
        f"chi2={chi2_majority:.3f}, df={majority_table.size - 1}, p={p_majority_chi:.4g}"
    )

    # Save key statistics for manual inspection if needed
    summary = {
        "age_p_social": age_p_social,
        "age_p_majority": age_p_majority,
        "site_p_social_lr": site_p_social,
        "site_p_majority_lr": site_p_majority,
        "site_p_social_chi2": p_social_chi,
        "site_p_majority_chi2": p_majority_chi,
    }
    with open("analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved key statistics to analysis_summary.json")


if __name__ == "__main__":
    main()

