import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def likelihood_ratio_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Reliance on social information: choosing either majority or minority option
    df["social"] = (df["y"] != 1).astype(int)

    # Preference for majority cues among children who relied on social information
    df["majority_choice"] = df["y"].map({2: 1, 3: 0})

    print("Basic sample sizes")
    print("Total N:", len(df))
    print("Social information N:", df["social"].sum())
    print("No social information N:", (df["social"] == 0).sum())
    print()

    # Descriptive statistics by culture and age
    social_by_culture = df.groupby("culture")["social"].mean()
    majority_by_culture = (
        df[df["social"] == 1].groupby("culture")["majority_choice"].mean()
    )

    print("Reliance on social information by culture (mean probability):")
    print(social_by_culture)
    print()
    print("Preference for majority option by culture (conditional on social; mean probability):")
    print(majority_by_culture)
    print()

    # Simple age bins for descriptive patterns
    bins = [4, 7, 10, 15]
    labels = ["4-6", "7-9", "10-14"]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    social_by_age = df.groupby("age_bin")["social"].mean()
    majority_by_age = (
        df[df["social"] == 1].groupby("age_bin")["majority_choice"].mean()
    )

    print("Reliance on social information by age bin:")
    print(social_by_age)
    print()
    print("Preference for majority option by age bin (conditional on social):")
    print(majority_by_age)
    print()

    # Logistic regression: social ~ age + culture
    model_social_full = smf.glm(
        "social ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_social_age_only = smf.glm(
        "social ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_social_culture_only = smf.glm(
        "social ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    lr_culture_social = likelihood_ratio_test(model_social_full, model_social_age_only)
    lr_age_social = likelihood_ratio_test(model_social_full, model_social_culture_only)

    print("Logistic regression for reliance on social information (social ~ age + culture)")
    print(model_social_full.summary())
    print("LRT for adding culture (vs age-only): LR=%.3f, df=%d, p=%.4g" % lr_culture_social)
    print("LRT for adding age (vs culture-only): LR=%.3f, df=%d, p=%.4g" % lr_age_social)
    print()

    # Logistic regression: majority_choice ~ age + culture (only where social=1)
    df_social = df[df["social"] == 1].copy()

    model_majority_full = smf.glm(
        "majority_choice ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    model_majority_age_only = smf.glm(
        "majority_choice ~ age",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    model_majority_culture_only = smf.glm(
        "majority_choice ~ C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    lr_culture_majority = likelihood_ratio_test(
        model_majority_full, model_majority_age_only
    )
    lr_age_majority = likelihood_ratio_test(
        model_majority_full, model_majority_culture_only
    )

    print("Logistic regression for preference for majority option (majority_choice ~ age + culture)")
    print(model_majority_full.summary())
    print(
        "LRT for adding culture (vs age-only): LR=%.3f, df=%d, p=%.4g"
        % lr_culture_majority
    )
    print(
        "LRT for adding age (vs culture-only): LR=%.3f, df=%d, p=%.4g"
        % lr_age_majority
    )
    print()

    # Effect size summaries
    print("Range of social-information use probability across cultures:")
    print("Min: %.3f, Max: %.3f" % (social_by_culture.min(), social_by_culture.max()))
    print("Range of majority-choice probability across cultures (conditional on social):")
    print(
        "Min: %.3f, Max: %.3f"
        % (majority_by_culture.min(), majority_by_culture.max())
    )
    print("Range of social-information use probability across age bins:")
    print("Min: %.3f, Max: %.3f" % (social_by_age.min(), social_by_age.max()))
    print("Range of majority-choice probability across age bins (conditional on social):")
    print("Min: %.3f, Max: %.3f" % (majority_by_age.min(), majority_by_age.max()))


if __name__ == "__main__":
    main()

