import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(model_full, model_reduced, df_diff=None):
    lr_stat = 2 * (model_full.llf - model_reduced.llf)
    if df_diff is None:
        df_diff = int(round(model_full.df_model - model_reduced.df_model))
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Construct key outcome variables
    df["social"] = (df["y"].isin([2, 3])).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("=== Basic description ===")
    print(f"N = {len(df)}")
    print("Outcome counts (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())
    print("\nProportion using social information (y in {2,3}):")
    print(df["social"].mean())
    print("\nProportion choosing the majority option overall:")
    print(df["majority_choice"].mean())

    print("\n=== By age (social information) ===")
    print(
        df.groupby("age")["social"]
        .mean()
        .reset_index()
        .rename(columns={"social": "p_social"})
    )

    print("\n=== By culture (social information) ===")
    print(
        df.groupby("culture")["social"]
        .mean()
        .reset_index()
        .rename(columns={"social": "p_social"})
    )

    df_social = df[df["social"] == 1].copy()
    print("\n=== By age (majority choice among social users) ===")
    print(
        df_social.groupby("age")["majority_choice"]
        .mean()
        .reset_index()
        .rename(columns={"majority_choice": "p_majority"})
    )

    print("\n=== By culture (majority choice among social users) ===")
    print(
        df_social.groupby("culture")["majority_choice"]
        .mean()
        .reset_index()
        .rename(columns={"majority_choice": "p_majority"})
    )

    # Logistic regressions for reliance on social information
    print("\n=== Logistic regression: social ~ age + culture ===")
    m_social_full = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    m_social_age_only = smf.logit("social ~ age", data=df).fit(disp=False)
    m_social_culture_only = smf.logit("social ~ C(culture)", data=df).fit(disp=False)

    print(m_social_full.summary())

    print("\nLikelihood-ratio tests for social information:")
    lr_culture_social = lr_test(m_social_full, m_social_age_only)
    lr_age_social = lr_test(m_social_full, m_social_culture_only)
    print(
        f"Culture effect (given age): LR={lr_culture_social[0]:.3f}, "
        f"df={lr_culture_social[1]}, p={lr_culture_social[2]:.5f}"
    )
    print(
        f"Age effect (given culture): LR={lr_age_social[0]:.3f}, "
        f"df={lr_age_social[1]}, p={lr_age_social[2]:.5f}"
    )

    # Logistic regressions for majority preference among social users
    print("\n=== Logistic regression: majority_choice ~ age + culture (social users only) ===")
    m_maj_full = smf.logit(
        "majority_choice ~ age + C(culture)", data=df_social
    ).fit(disp=False)
    m_maj_age_only = smf.logit("majority_choice ~ age", data=df_social).fit(disp=False)
    m_maj_culture_only = smf.logit(
        "majority_choice ~ C(culture)", data=df_social
    ).fit(disp=False)

    print(m_maj_full.summary())

    print("\nLikelihood-ratio tests for majority preference:")
    lr_culture_maj = lr_test(m_maj_full, m_maj_age_only)
    lr_age_maj = lr_test(m_maj_full, m_maj_culture_only)
    print(
        f"Culture effect (given age): LR={lr_culture_maj[0]:.3f}, "
        f"df={lr_culture_maj[1]}, p={lr_culture_maj[2]:.5f}"
    )
    print(
        f"Age effect (given culture): LR={lr_age_maj[0]:.3f}, "
        f"df={lr_age_maj[1]}, p={lr_age_maj[2]:.5f}"
    )

    print("\n=== Chi-square test: social by age (categorical age) ===")
    ct_social_age = pd.crosstab(df["age"], df["social"])
    print(ct_social_age)
    chi2_sa, p_sa, dof_sa, _ = stats.chi2_contingency(ct_social_age)
    print(f"Chi2={chi2_sa:.3f}, df={dof_sa}, p={p_sa:.5f}")

    print("\n=== Chi-square test: majority_choice by age (social users only) ===")
    ct_maj_age = pd.crosstab(df_social["age"], df_social["majority_choice"])
    print(ct_maj_age)
    chi2_ma, p_ma, dof_ma, _ = stats.chi2_contingency(ct_maj_age)
    print(f"Chi2={chi2_ma:.3f}, df={dof_ma}, p={p_ma:.5f}")


if __name__ == "__main__":
    main()
