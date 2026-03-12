import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def load_data(path: str = "boxes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Clarify variable meanings based on info.json description.
    # majority_first: 1=unchosen, 2=majority, 3=minority (outcome)
    # gender: 1=girl, 2=boy
    # age: 4-14 (years)
    # culture: actually indicates whether the majority option was demonstrated first (0/1)
    # y: site ID (1-8), used as a proxy for cultural context
    df = df.rename(
        columns={
            "majority_first": "outcome",
            "culture": "majority_demo_first",
            "y": "site",
        }
    )
    # Derived variables
    df["is_social"] = df["outcome"].isin([2, 3]).astype(int)
    df["is_majority"] = (df["outcome"] == 2).astype(int)

    # Age groups to represent developmental stages
    def age_group(a: float) -> str:
        if a <= 6:
            return "4-6"
        if a <= 9:
            return "7-9"
        if a <= 12:
            return "10-12"
        return "13-14"

    df["age_group"] = df["age"].apply(age_group)
    return df


def chi_square_table(df: pd.DataFrame, row: str, col: str):
    table = pd.crosstab(df[row], df[col])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "table": table,
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
    }


def logistic_lr_test(formula_reduced: str, formula_full: str, data: pd.DataFrame):
    reduced = smf.logit(formula_reduced, data=data).fit(disp=False)
    full = smf.logit(formula_full, data=data).fit(disp=False)
    # Manual likelihood-ratio test: 2 * (LL_full - LL_reduced)
    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return {
        "lr_stat": lr_stat,
        "p": p_value,
        "df_diff": int(df_diff),
        "full_params": full.params,
    }


def main():
    df = load_data()
    n = len(df)
    print(f"Number of children: {n}")

    # Basic outcome distribution
    outcome_counts = df["outcome"].value_counts().sort_index()
    print("\nOutcome distribution (1=unchosen, 2=majority, 3=minority):")
    print(outcome_counts)
    print("Proportions:")
    print((outcome_counts / n).round(3))

    # Social learning (any demonstrated option) vs age group and site
    print("\n=== Social learning (is_social) x age_group ===")
    social_age = chi_square_table(df, "age_group", "is_social")
    print(social_age["table"])
    print(
        f"Chi2={social_age['chi2']:.3f}, dof={social_age['dof']}, "
        f"p={social_age['p']:.5f}"
    )

    print("\n=== Social learning (is_social) x site ===")
    social_site = chi_square_table(df, "site", "is_social")
    print(social_site["table"])
    print(
        f"Chi2={social_site['chi2']:.3f}, dof={social_site['dof']}, "
        f"p={social_site['p']:.5f}"
    )

    # Majority preference among social learners
    social_df = df[df["is_social"] == 1].copy()
    print(f"\nNumber of social learners: {len(social_df)}")

    print("\n=== Majority choice (is_majority) x age_group among social learners ===")
    maj_age = chi_square_table(social_df, "age_group", "is_majority")
    print(maj_age["table"])
    print(
        f"Chi2={maj_age['chi2']:.3f}, dof={maj_age['dof']}, "
        f"p={maj_age['p']:.5f}"
    )

    print("\n=== Majority choice (is_majority) x site among social learners ===")
    maj_site = chi_square_table(social_df, "site", "is_majority")
    print(maj_site["table"])
    print(
        f"Chi2={maj_site['chi2']:.3f}, dof={maj_site['dof']}, "
        f"p={maj_site['p']:.5f}"
    )

    # Logistic regression: does social learning vary with continuous age and site?
    print("\n=== Logistic regression: is_social ~ age + site (LR test for site) ===")
    lr_social_site = logistic_lr_test("is_social ~ age", "is_social ~ age + C(site)", df)
    print(
        f"LR stat={lr_social_site['lr_stat']:.3f}, df={lr_social_site['df_diff']}, "
        f"p={lr_social_site['p']:.5f}"
    )
    print("Full model coefficients:")
    print(lr_social_site["full_params"])

    # Logistic regression: majority preference vs age and site among social learners
    print("\n=== Logistic regression: is_majority ~ age + site (social learners only) ===")
    lr_maj_site = logistic_lr_test(
        "is_majority ~ age", "is_majority ~ age + C(site)", social_df
    )
    print(
        f"LR stat={lr_maj_site['lr_stat']:.3f}, df={lr_maj_site['df_diff']}, "
        f"p={lr_maj_site['p']:.5f}"
    )
    print("Full model coefficients:")
    print(lr_maj_site["full_params"])

    # Additional: direct age effects (developmental) without site controls
    print("\n=== Logistic regression: is_social ~ age (developmental effect) ===")
    social_age_model = smf.logit("is_social ~ age", data=df).fit(disp=False)
    print("Coefficients:")
    print(social_age_model.params)
    print("P-values:")
    print(social_age_model.pvalues)

    print("\n=== Logistic regression: is_majority ~ age (social learners only) ===")
    maj_age_model = smf.logit("is_majority ~ age", data=social_df).fit(disp=False)
    print("Coefficients:")
    print(maj_age_model.params)
    print("P-values:")
    print(maj_age_model.pvalues)


if __name__ == "__main__":
    main()
