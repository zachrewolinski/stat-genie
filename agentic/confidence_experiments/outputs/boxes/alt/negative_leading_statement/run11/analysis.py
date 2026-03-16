import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df_social = df[df["y"].isin([2, 3])].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    print(f"Total N: {len(df)}")
    print("\nOutcome (y) counts (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())

    print("\nSocial vs asocial choices (social = majority or minority option):")
    print(df["social"].value_counts())

    print("\nMean social reliance by age:")
    print(df.groupby("age")["social"].mean())

    print("\nMean social reliance by culture:")
    print(df.groupby("culture")["social"].mean())

    print("\nMajority preference among social learners by age:")
    print(df_social.groupby("age")["majority_choice"].mean())

    print("\nMajority preference among social learners by culture:")
    print(df_social.groupby("culture")["majority_choice"].mean())

    # Center age for regression
    df["age_c"] = df["age"] - df["age"].mean()
    df_social["age_c"] = df_social["age"] - df_social["age"].mean()

    # Logistic regression for reliance on social information
    print(
        "\nLikelihood-ratio tests for social reliance "
        "(social ~ age + culture + majority_first + gender):"
    )
    model_social = smf.logit(
        "social ~ age_c + C(culture) + majority_first + gender", data=df
    ).fit(disp=False, maxiter=200)
    model_social_no_age = smf.logit(
        "social ~ C(culture) + majority_first + gender", data=df
    ).fit(disp=False, maxiter=200)
    model_social_no_culture = smf.logit(
        "social ~ age_c + majority_first + gender", data=df
    ).fit(disp=False, maxiter=200)

    lr_test(model_social, model_social_no_age, "Effect of age on social reliance")
    lr_test(model_social, model_social_no_culture, "Effect of culture on social reliance")

    # Logistic regression for majority preference among social learners
    print(
        "\nLikelihood-ratio tests for majority preference "
        "(majority_choice ~ age + culture + majority_first + gender, social-only):"
    )
    model_majority = smf.logit(
        "majority_choice ~ age_c + C(culture) + majority_first + gender",
        data=df_social,
    ).fit(disp=False, maxiter=200)
    model_majority_no_age = smf.logit(
        "majority_choice ~ C(culture) + majority_first + gender", data=df_social
    ).fit(disp=False, maxiter=200)
    model_majority_no_culture = smf.logit(
        "majority_choice ~ age_c + majority_first + gender", data=df_social
    ).fit(disp=False, maxiter=200)

    lr_test(
        model_majority,
        model_majority_no_age,
        "Effect of age on majority preference",
    )
    lr_test(
        model_majority,
        model_majority_no_culture,
        "Effect of culture on majority preference",
    )


def lr_test(full_model, reduced_model, label: str):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_val = stats.chi2.sf(lr_stat, df_diff)
    print(f"{label}: LR={lr_stat:.3f}, df={df_diff}, p={p_val:.4g}")
    return lr_stat, df_diff, p_val


if __name__ == "__main__":
    main()

