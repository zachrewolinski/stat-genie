import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def fit_logit(formula, data):
    """Fit a logistic regression model and return the fitted result."""
    model = smf.logit(formula=formula, data=data)
    result = model.fit(disp=False)
    return result


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test comparing a full and reduced nested model."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Derived variables reflecting social learning outcomes
    df["social_reliance"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["age_c"] = df["age"] - df["age"].mean()

    # Age groups (rough developmental stages)
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    print("=== Sample size and basic outcome distribution ===")
    print(f"N = {len(df)}")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    # Descriptive cross-tabs for social reliance and majority preference
    print("=== Social reliance (demonstrated option vs undemonstrated) by culture ===")
    ct_culture_reliance = pd.crosstab(df["culture"], df["social_reliance"])
    print(ct_culture_reliance)
    chi2_cult_rely, p_cult_rely, dof_cult_rely, _ = stats.chi2_contingency(ct_culture_reliance)
    print(f"Chi2={chi2_cult_rely:.3f}, df={dof_cult_rely}, p={p_cult_rely:.4f}")
    print()

    print("=== Social reliance by age group ===")
    ct_age_reliance = pd.crosstab(df["age_group"], df["social_reliance"])
    print(ct_age_reliance)
    chi2_age_rely, p_age_rely, dof_age_rely, _ = stats.chi2_contingency(ct_age_reliance)
    print(f"Chi2={chi2_age_rely:.3f}, df={dof_age_rely}, p={p_age_rely:.4f}")
    print()

    demo_mask = df["y"].isin([2, 3])
    df_demo = df.loc[demo_mask].copy()

    print("=== Majority vs minority choice (among demonstrators) by culture ===")
    ct_culture_majority = pd.crosstab(df_demo["culture"], df_demo["majority_choice"])
    print(ct_culture_majority)
    chi2_cult_maj, p_cult_maj, dof_cult_maj, _ = stats.chi2_contingency(ct_culture_majority)
    print(f"Chi2={chi2_cult_maj:.3f}, df={dof_cult_maj}, p={p_cult_maj:.4f}")
    print()

    print("=== Majority vs minority choice by age group ===")
    ct_age_majority = pd.crosstab(df_demo["age_group"], df_demo["majority_choice"])
    print(ct_age_majority)
    chi2_age_maj, p_age_maj, dof_age_maj, _ = stats.chi2_contingency(ct_age_majority)
    print(f"Chi2={chi2_age_maj:.3f}, df={dof_age_maj}, p={p_age_maj:.4f}")
    print()

    # Logistic regression: social reliance ~ age + culture + controls
    print("=== Logistic regression: social_reliance ~ age + culture + gender + majority_first ===")
    m1_full = fit_logit("social_reliance ~ age_c + C(culture) + gender + majority_first", df)
    print(m1_full.summary())
    print()

    # LR tests for age and culture effects on social reliance
    m1_no_age = fit_logit("social_reliance ~ C(culture) + gender + majority_first", df)
    lr_age1, df_age1, p_age1 = lr_test(m1_full, m1_no_age)
    print(f"LR test for age effect on social_reliance: LR={lr_age1:.3f}, df={df_age1}, p={p_age1:.4f}")

    m1_no_culture = fit_logit("social_reliance ~ age_c + gender + majority_first", df)
    lr_cult1, df_cult1, p_cult1 = lr_test(m1_full, m1_no_culture)
    print(f"LR test for culture effect on social_reliance: LR={lr_cult1:.3f}, df={df_cult1}, p={p_cult1:.4f}")
    print()

    # Logistic regression: majority preference among those who followed any demonstrator
    print("=== Logistic regression: majority_choice ~ age + culture + gender + majority_first (demonstrator choices only) ===")
    m2_full = fit_logit("majority_choice ~ age_c + C(culture) + gender + majority_first", df_demo)
    print(m2_full.summary())
    print()

    m2_no_age = fit_logit("majority_choice ~ C(culture) + gender + majority_first", df_demo)
    lr_age2, df_age2, p_age2 = lr_test(m2_full, m2_no_age)
    print(f"LR test for age effect on majority_choice: LR={lr_age2:.3f}, df={df_age2}, p={p_age2:.4f}")

    m2_no_culture = fit_logit("majority_choice ~ age_c + gender + majority_first", df_demo)
    lr_cult2, df_cult2, p_cult2 = lr_test(m2_full, m2_no_culture)
    print(f"LR test for culture effect on majority_choice: LR={lr_cult2:.3f}, df={df_cult2}, p={p_cult2:.4f}")
    print()


if __name__ == "__main__":
    main()

