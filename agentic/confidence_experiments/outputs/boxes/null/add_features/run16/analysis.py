import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def lr_test(full_model, reduced_model):
    """Likelihood ratio test comparing two nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = int(full_model.df_model - reduced_model.df_model)
    p_value = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Define social-information use: 1 if child followed any demonstrated option (majority or minority)
    df["social"] = df["y"].isin([2, 3]).astype(int)

    # Subset of trials where social information was used
    df_social = df[df["social"] == 1].copy()
    df_social["majority"] = (df_social["y"] == 2).astype(int)

    # Age groups from metadata description
    age_map = {
        17.5: "under_20",
        22.0: "20_24",
        27.0: "25_29",
        32.0: "30_34",
        37.0: "35_39",
        42.0: "40_44",
        47.0: "45_49",
        52.0: "50_54",
        57.0: "55_plus",
    }
    df["age_group"] = df["age"].map(age_map)
    df["age_group"] = df["age_group"].astype("category")
    df_social["age_group"] = df_social["age"].map(age_map)
    df_social["age_group"] = df_social["age_group"].astype("category")

    # Descriptive statistics
    overall_social = df["social"].mean()
    overall_majority_given_social = df_social["majority"].mean()

    by_culture_social = df.groupby("culture")["social"].mean()
    by_culture_majority = df_social.groupby("culture")["majority"].mean()

    by_age_social = df.groupby("age_group")["social"].mean()
    by_age_majority = df_social.groupby("age_group")["majority"].mean()

    print("Overall social reliance (P(y in {{2,3}})): {:.3f}".format(overall_social))
    print(
        "Overall majority choice given social (P(y=2 | y in {{2,3}})): {:.3f}".format(
            overall_majority_given_social
        )
    )

    print("\nSocial reliance by culture:")
    print(by_culture_social)

    print("\nMajority preference by culture:")
    print(by_culture_majority)

    print("\nSocial reliance by age_group:")
    print(by_age_social)

    print("\nMajority preference by age_group:")
    print(by_age_majority)

    # Logistic regression: social information use ~ culture + age_group
    model_social_full = smf.logit("social ~ C(culture) + C(age_group)", data=df).fit(
        disp=False, maxiter=1000
    )
    model_social_culture_only = smf.logit("social ~ C(culture)", data=df).fit(
        disp=False, maxiter=1000
    )
    model_social_age_only = smf.logit("social ~ C(age_group)", data=df).fit(
        disp=False, maxiter=1000
    )

    lr_culture_social = lr_test(model_social_full, model_social_age_only)
    lr_age_social = lr_test(model_social_full, model_social_culture_only)

    print("\nLikelihood-ratio tests for social-information use:")
    print(
        "  Culture effect (controlling for age_group): LR={:.3f}, df={}, p={:.4g}".format(
            lr_culture_social[0], lr_culture_social[1], lr_culture_social[2]
        )
    )
    print(
        "  Age_group effect (controlling for culture): LR={:.3f}, df={}, p={:.4g}".format(
            lr_age_social[0], lr_age_social[1], lr_age_social[2]
        )
    )

    # Logistic regression: majority vs minority choice among social learners
    model_majority_full = smf.logit(
        "majority ~ C(culture) + C(age_group)", data=df_social
    ).fit(disp=False, maxiter=1000)
    model_majority_culture_only = smf.logit(
        "majority ~ C(culture)", data=df_social
    ).fit(disp=False, maxiter=1000)
    model_majority_age_only = smf.logit(
        "majority ~ C(age_group)", data=df_social
    ).fit(disp=False, maxiter=1000)

    lr_culture_majority = lr_test(model_majority_full, model_majority_age_only)
    lr_age_majority = lr_test(model_majority_full, model_majority_culture_only)

    print("\nLikelihood-ratio tests for majority vs minority choice (among social users):")
    print(
        "  Culture effect (controlling for age_group): LR={:.3f}, df={}, p={:.4g}".format(
            lr_culture_majority[0], lr_culture_majority[1], lr_culture_majority[2]
        )
    )
    print(
        "  Age_group effect (controlling for culture): LR={:.3f}, df={}, p={:.4g}".format(
            lr_age_majority[0], lr_age_majority[1], lr_age_majority[2]
        )
    )


if __name__ == "__main__":
    main()
