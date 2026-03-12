import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")
    print("Basic info")
    print(f"N = {len(df)}")
    print(df.head())

    # Recode outcomes
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority"] = (df["y"] == 2).astype(int)

    # Define coarse age groups for descriptive tables
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    print("\nOutcome distribution (y)")
    print(df["y"].value_counts().sort_index())

    print("\nOutcome by culture (rows=culture, cols=y)")
    tab_culture_y = pd.crosstab(df["culture"], df["y"])
    print(tab_culture_y)

    print("\nOutcome by age group (rows=age_group, cols=y)")
    tab_age_y = pd.crosstab(df["age_group"], df["y"])
    print(tab_age_y)

    # Chi-square tests for association of outcome with culture and age_group
    chi2_cult, p_cult, dof_cult, _ = stats.chi2_contingency(tab_culture_y)
    print(
        f"\nChi-square test of y by culture: "
        f"chi2 = {chi2_cult:.3f}, df = {dof_cult}, p = {p_cult:.4g}"
    )

    chi2_age, p_age, dof_age, _ = stats.chi2_contingency(tab_age_y)
    print(
        f"Chi-square test of y by age_group: "
        f"chi2 = {chi2_age:.3f}, df = {dof_age}, p = {p_age:.4g}"
    )

    # Social-information reliance: choosing any demonstrated option vs undemonstrated
    print("\n=== Logistic regression: social-information reliance (social vs undemonstrated) ===")
    model_social = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    print(model_social.summary())

    model_social_no_cult = smf.logit("social ~ age", data=df).fit(disp=False)
    lr_stat_cult = 2 * (model_social.llf - model_social_no_cult.llf)
    df_cult = model_social.df_model - model_social_no_cult.df_model
    p_lr_cult = stats.chi2.sf(lr_stat_cult, df_cult)
    print(
        f"\nLR test: adding culture to age-only model for social "
        f"(tests culture effect on social reliance): "
        f"chi2 = {lr_stat_cult:.3f}, df = {df_cult}, p = {p_lr_cult:.4g}"
    )

    model_social_no_age = smf.logit("social ~ C(culture)", data=df).fit(disp=False)
    lr_stat_age = 2 * (model_social.llf - model_social_no_age.llf)
    df_age_lr = model_social.df_model - model_social_no_age.df_model
    p_lr_age = stats.chi2.sf(lr_stat_age, df_age_lr)
    print(
        f"LR test: adding age to culture-only model for social "
        f"(tests age effect on social reliance): "
        f"chi2 = {lr_stat_age:.3f}, df = {df_age_lr}, p = {p_lr_age:.4g}"
    )

    # Majority preference among those who use social information
    df_social = df[df["social"] == 1].copy()
    print(
        f"\nSubsample using social information (y in {{2,3}}): "
        f"N = {len(df_social)}"
    )
    print("Majority vs minority choices overall:")
    print(df_social["majority"].value_counts())

    print("\nMajority vs minority by culture")
    tab_culture_maj = pd.crosstab(df_social["culture"], df_social["majority"])
    print(tab_culture_maj)

    print("\nMajority vs minority by age_group")
    tab_age_maj = pd.crosstab(df_social["age_group"], df_social["majority"])
    print(tab_age_maj)

    chi2_cult_maj, p_cult_maj, dof_cult_maj, _ = stats.chi2_contingency(tab_culture_maj)
    print(
        f"\nChi-square test of majority vs culture: "
        f"chi2 = {chi2_cult_maj:.3f}, df = {dof_cult_maj}, p = {p_cult_maj:.4g}"
    )

    chi2_age_maj, p_age_maj, dof_age_maj, _ = stats.chi2_contingency(tab_age_maj)
    print(
        f"Chi-square test of majority vs age_group: "
        f"chi2 = {chi2_age_maj:.3f}, df = {dof_age_maj}, p = {p_age_maj:.4g}"
    )

    print("\n=== Logistic regression: majority preference among social learners ===")
    model_maj = smf.logit("majority ~ age + C(culture)", data=df_social).fit(disp=False)
    print(model_maj.summary())

    model_maj_no_cult = smf.logit("majority ~ age", data=df_social).fit(disp=False)
    lr_stat_cult_maj = 2 * (model_maj.llf - model_maj_no_cult.llf)
    df_cult_maj = model_maj.df_model - model_maj_no_cult.df_model
    p_lr_cult_maj = stats.chi2.sf(lr_stat_cult_maj, df_cult_maj)
    print(
        f"\nLR test: adding culture to age-only model for majority preference: "
        f"chi2 = {lr_stat_cult_maj:.3f}, df = {df_cult_maj}, p = {p_lr_cult_maj:.4g}"
    )

    model_maj_no_age = smf.logit("majority ~ C(culture)", data=df_social).fit(disp=False)
    lr_stat_age_maj = 2 * (model_maj.llf - model_maj_no_age.llf)
    df_age_maj_lr = model_maj.df_model - model_maj_no_age.df_model
    p_lr_age_maj = stats.chi2.sf(lr_stat_age_maj, df_age_maj_lr)
    print(
        f"LR test: adding age to culture-only model for majority preference: "
        f"chi2 = {lr_stat_age_maj:.3f}, df = {df_age_maj_lr}, p = {p_lr_age_maj:.4g}"
    )


if __name__ == "__main__":
    main()

