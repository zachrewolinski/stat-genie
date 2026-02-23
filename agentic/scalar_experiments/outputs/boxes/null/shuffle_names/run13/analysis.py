import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, chi2_contingency


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # 1 = undemonstrated option, 2 = majority, 3 = minority
    df["social"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Basic descriptives
    n = len(df)
    social_rate = df["social"].mean()
    majority_rate_overall = (df["majority_first"] == 2).mean()
    print(f"Total N: {n}")
    print(f"Overall social-reliance rate (any demonstrated option): {social_rate:.3f}")
    print(f"Overall majority-choice rate (of all trials): {majority_rate_overall:.3f}")

    # Age groups for descriptive purposes
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    print("\nSocial-reliance rate by age group:")
    print(
        df.groupby("age_group")["social"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "social_mean", "count": "n"})
    )

    print("\nMajority-choice rate by age group (all children):")
    print(
        df.groupby("age_group")["majority_choice"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "majority_mean", "count": "n"})
    )

    # Treat y as site/culture ID
    print("\nSocial-reliance rate by site (y):")
    print(
        df.groupby("y")["social"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "social_mean", "count": "n"})
    )

    print("\nMajority-choice rate by site (y):")
    print(
        df.groupby("y")["majority_choice"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "majority_mean", "count": "n"})
    )

    # Helper for likelihood-ratio tests
    def lr_test(full_model, reduced_model):
        lr_stat = 2 * (full_model.llf - reduced_model.llf)
        df_diff = full_model.df_model - reduced_model.df_model
        p_value = chi2.sf(lr_stat, df_diff)
        return lr_stat, df_diff, p_value

    # Logistic regression: social reliance ~ age + site
    social_full = smf.logit("social ~ age + C(y)", data=df).fit(disp=False)
    social_age_only = smf.logit("social ~ age", data=df).fit(disp=False)
    social_site_only = smf.logit("social ~ C(y)", data=df).fit(disp=False)

    print("\n=== Logistic regression: social ~ age + C(y) ===")
    print(social_full.summary())
    lr_site_stat, lr_site_df, lr_site_p = lr_test(social_full, social_age_only)
    lr_age_stat, lr_age_df, lr_age_p = lr_test(social_full, social_site_only)
    print(
        f"\nLR test for added site effect (C(y)) over age-only model: "
        f"LR={lr_site_stat:.3f}, df={lr_site_df}, p={lr_site_p:.4g}"
    )
    print(
        f"LR test for added age effect over site-only model: "
        f"LR={lr_age_stat:.3f}, df={lr_age_df}, p={lr_age_p:.4g}"
    )
    print(
        f"Age coefficient in full social model: "
        f"beta={social_full.params['age']:.3f}, p={social_full.pvalues['age']:.4g}"
    )

    # Logistic regression: majority preference among socially guided choices
    df_social = df[df["social"] == 1].copy()
    majority_full = smf.logit("majority_choice ~ age + C(y)", data=df_social).fit(
        disp=False
    )
    majority_age_only = smf.logit("majority_choice ~ age", data=df_social).fit(
        disp=False
    )
    majority_site_only = smf.logit("majority_choice ~ C(y)", data=df_social).fit(
        disp=False
    )

    print("\n=== Logistic regression: majority_choice ~ age + C(y) (social choices only) ===")
    print(majority_full.summary())
    lr_site_stat_m, lr_site_df_m, lr_site_p_m = lr_test(majority_full, majority_age_only)
    lr_age_stat_m, lr_age_df_m, lr_age_p_m = lr_test(majority_full, majority_site_only)
    print(
        f"\nLR test for added site effect (C(y)) over age-only model: "
        f"LR={lr_site_stat_m:.3f}, df={lr_site_df_m}, p={lr_site_p_m:.4g}"
    )
    print(
        f"LR test for added age effect over site-only model: "
        f"LR={lr_age_stat_m:.3f}, df={lr_age_df_m}, p={lr_age_p_m:.4g}"
    )
    print(
        f"Age coefficient in full majority model: "
        f"beta={majority_full.params['age']:.3f}, p={majority_full.pvalues['age']:.4g}"
    )

    # Chi-square tests using categorical age groups and sites
    social_age_table = pd.crosstab(df["age_group"], df["social"])
    chi2_sa, p_sa, dof_sa, _ = chi2_contingency(social_age_table)
    print(
        "\nChi-square test of social reliance vs age_group: "
        f"chi2={chi2_sa:.3f}, df={dof_sa}, p={p_sa:.4g}"
    )

    majority_age_table = pd.crosstab(df["age_group"], df["majority_choice"])
    chi2_ma, p_ma, dof_ma, _ = chi2_contingency(majority_age_table)
    print(
        "Chi-square test of majority choice vs age_group: "
        f"chi2={chi2_ma:.3f}, df={dof_ma}, p={p_ma:.4g}"
    )

    social_site_table = pd.crosstab(df["y"], df["social"])
    chi2_ss, p_ss, dof_ss, _ = chi2_contingency(social_site_table)
    print(
        "Chi-square test of social reliance vs site (y): "
        f"chi2={chi2_ss:.3f}, df={dof_ss}, p={p_ss:.4g}"
    )

    majority_site_table = pd.crosstab(df_social["y"], df_social["majority_choice"])
    chi2_ms, p_ms, dof_ms, _ = chi2_contingency(majority_site_table)
    print(
        "Chi-square test of majority choice vs site (y): "
        f"chi2={chi2_ms:.3f}, df={dof_ms}, p={p_ms:.4g}"
    )


if __name__ == "__main__":
    main()
