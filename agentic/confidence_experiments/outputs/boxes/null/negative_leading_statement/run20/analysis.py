import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full_model, reduced_model, name: str) -> None:
    """Compute and print a likelihood-ratio test comparing two nested models."""
    ll_full = full_model.llf
    ll_reduced = reduced_model.llf
    df_full = full_model.df_model
    df_reduced = reduced_model.df_model
    df_diff = df_full - df_reduced
    lr_stat = 2 * (ll_full - ll_reduced)
    p_value = 1 - chi2.cdf(lr_stat, df_diff)
    print(f"{name}: stat={lr_stat:.3f}, df={df_diff:.0f}, p={p_value:.4g}")


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["social"] = (df["y"] != 1).astype(int)  # 1 = followed any demonstrator
    df["majority_choice"] = (df["y"] == 2).astype(int)  # 1 = followed majority demonstrators

    # Age groups for descriptive summaries
    bins = [4, 7, 10, 15]
    labels = ["4-6", "7-9", "10-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    print("N =", len(df))
    print("\nOverall outcome proportions (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nReliance on social information (P(social) = P(y != 1)) by culture:")
    print(
        df.groupby("culture")["social"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "p_social", "count": "n"})
    )

    print("\nReliance on social information by age group:")
    print(
        df.groupby("age_group")["social"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "p_social", "count": "n"})
    )

    df_social = df[df["social"] == 1].copy()
    print("\nAmong social choices, preference for majority (P(majority_choice)) by culture:")
    print(
        df_social.groupby("culture")["majority_choice"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "p_majority", "count": "n"})
    )

    print("\nAmong social choices, preference for majority by age group:")
    print(
        df_social.groupby("age_group")["majority_choice"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "p_majority", "count": "n"})
    )

    # Logistic regression: reliance on social information
    print("\n=== Logistic regression: social ~ age + culture + gender + majority_first ===")
    formula_social_full = "social ~ age + C(culture) + gender + majority_first"
    model_social_full = smf.logit(formula_social_full, data=df).fit(disp=False)
    print(model_social_full.summary())

    # Likelihood-ratio tests for age and culture
    model_social_no_age = smf.logit(
        "social ~ C(culture) + gender + majority_first", data=df
    ).fit(disp=False)
    model_social_no_culture = smf.logit(
        "social ~ age + gender + majority_first", data=df
    ).fit(disp=False)

    print("\nLikelihood-ratio tests for social model:")
    lr_test(
        model_social_full,
        model_social_no_age,
        "  Age effect on social reliance",
    )
    lr_test(
        model_social_full,
        model_social_no_culture,
        "  Culture effect on social reliance",
    )

    # Logistic regression: preference for majority among social choices
    print(
        "\n=== Logistic regression: majority_choice ~ age + culture + gender + majority_first "
        "(restricted to social choices) ==="
    )
    formula_maj_full = "majority_choice ~ age + C(culture) + gender + majority_first"
    model_maj_full = smf.logit(formula_maj_full, data=df_social).fit(disp=False)
    print(model_maj_full.summary())

    model_maj_no_age = smf.logit(
        "majority_choice ~ C(culture) + gender + majority_first", data=df_social
    ).fit(disp=False)
    model_maj_no_culture = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=df_social
    ).fit(disp=False)

    print("\nLikelihood-ratio tests for majority-choice model:")
    lr_test(
        model_maj_full,
        model_maj_no_age,
        "  Age effect on majority preference",
    )
    lr_test(
        model_maj_full,
        model_maj_no_culture,
        "  Culture effect on majority preference",
    )


if __name__ == "__main__":
    main()
