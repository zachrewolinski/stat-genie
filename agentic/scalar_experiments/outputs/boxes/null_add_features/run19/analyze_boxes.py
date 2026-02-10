import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def summarize_rates(df: pd.DataFrame) -> None:
    df = df.copy()
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    print("Overall counts (y):")
    print(df["y"].value_counts().sort_index())
    print("\nOverall social rate (choose majority or minority):")
    print(df["social"].mean())

    print("\nOverall majority among social choices:")
    print(df_social["majority_choice"].mean())

    # Age quartiles as crude developmental stages
    q1, q3 = df["age"].quantile([0.25, 0.75])
    young = df[df["age"] <= q1]
    old = df[df["age"] >= q3]

    print(f"\nAge quartiles: Q1={q1:.2f}, Q3={q3:.2f}")
    print("Social rate by age group (young vs old):")
    for label, subset in [("young", young), ("old", old)]:
        print(label, subset["social"].mean(), len(subset))

    young_social = df_social[df_social["age"] <= q1]
    old_social = df_social[df_social["age"] >= q3]
    print("\nMajority preference among social choices by age group:")
    for label, subset in [("young", young_social), ("old", old_social)]:
        print(label, subset["majority_choice"].mean(), len(subset))

    # By culture
    print("\nSocial rate by culture:")
    social_by_culture = df.groupby("culture")["social"].mean()
    print(social_by_culture)
    print("Range (max - min):", social_by_culture.max() - social_by_culture.min())

    print("\nMajority preference among social choices by culture:")
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()
    print(majority_by_culture)
    print("Range (max - min):", majority_by_culture.max() - majority_by_culture.min())


def run_logistic_models(df: pd.DataFrame) -> None:
    df = df.copy()
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    print("\n=== Logistic regression: social ~ age + culture ===")
    m_full = smf.logit("social ~ age + C(culture)", data=df).fit(disp=0)
    m_age_only = smf.logit("social ~ age", data=df).fit(disp=0)
    m_intercept = smf.logit("social ~ 1", data=df).fit(disp=0)

    print(m_full.summary())
    print("\nP-value for age (social):", m_full.pvalues["age"])

    # LR test for culture (full vs age-only)
    lr_culture = 2 * (m_full.llf - m_age_only.llf)
    df_culture = int(m_full.df_model - m_age_only.df_model)
    p_culture = stats.chi2.sf(lr_culture, df_culture)
    print(
        f"LR test for culture in social model: LR={lr_culture:.3f}, "
        f"df={df_culture}, p={p_culture:.5f}"
    )

    # LR test for age (age-only vs intercept)
    lr_age = 2 * (m_age_only.llf - m_intercept.llf)
    df_age = int(m_age_only.df_model - m_intercept.df_model)
    p_age_lr = stats.chi2.sf(lr_age, df_age)
    print(
        f"LR test for age in social model: LR={lr_age:.3f}, "
        f"df={df_age}, p={p_age_lr:.5f}"
    )

    print("\n=== Logistic regression: majority_choice ~ age + culture (social only) ===")
    m_full_maj = smf.logit("majority_choice ~ age + C(culture)", data=df_social).fit(
        disp=0
    )
    m_age_only_maj = smf.logit("majority_choice ~ age", data=df_social).fit(disp=0)
    m_intercept_maj = smf.logit("majority_choice ~ 1", data=df_social).fit(disp=0)

    print(m_full_maj.summary())
    print("\nP-value for age (majority):", m_full_maj.pvalues["age"])

    lr_culture_maj = 2 * (m_full_maj.llf - m_age_only_maj.llf)
    df_culture_maj = int(m_full_maj.df_model - m_age_only_maj.df_model)
    p_culture_maj = stats.chi2.sf(lr_culture_maj, df_culture_maj)
    print(
        f"LR test for culture in majority model: LR={lr_culture_maj:.3f}, "
        f"df={df_culture_maj}, p={p_culture_maj:.5f}"
    )

    lr_age_maj = 2 * (m_age_only_maj.llf - m_intercept_maj.llf)
    df_age_maj = int(m_age_only_maj.df_model - m_intercept_maj.df_model)
    p_age_lr_maj = stats.chi2.sf(lr_age_maj, df_age_maj)
    print(
        f"LR test for age in majority model: LR={lr_age_maj:.3f}, "
        f"df={df_age_maj}, p={p_age_lr_maj:.5f}"
    )


def main() -> None:
    df = pd.read_csv("boxes.csv")
    summarize_rates(df)
    run_logistic_models(df)


if __name__ == "__main__":
    main()

