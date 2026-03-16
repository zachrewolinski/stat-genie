import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(reduced_model, full_model):
    """Likelihood ratio test comparing two nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["social"] = (df["y"] != 1).astype(int)  # 1 if child followed any demonstrator
    df["majority"] = (df["y"] == 2).astype(int)  # 1 if child followed majority choice

    print("N observations:", len(df))

    print("\nOverall outcome proportions (y):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nOutcome proportions by culture:")
    print(
        df.groupby("culture")["y"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
        .sort_index()
    )

    print("\nMean age by culture:")
    print(
        df.groupby("culture")["age"].agg(["mean", "std", "min", "max"]).round(2)
    )

    print("\nSocial choice proportion by culture:")
    print(df.groupby("culture")["social"].mean().round(3))

    print("\nMajority choice proportion by culture:")
    print(df.groupby("culture")["majority"].mean().round(3))

    # Models for reliance on social information (social vs non-social)
    print("\n=== Logistic models: reliance on social information (social) ===")
    try:
        social_base = smf.logit(
            "social ~ age + majority_first", data=df
        ).fit(disp=False)
        print("\nBase model social ~ age + majority_first")
        print(social_base.summary())

        social_culture = smf.logit(
            "social ~ age + majority_first + C(culture)", data=df
        ).fit(disp=False)
        print("\nModel with culture: social ~ age + majority_first + C(culture)")
        print(social_culture.summary())

        lr_stat, df_diff, p_val = lr_test(social_base, social_culture)
        print(
            f"\nLR test for adding culture to social model: "
            f"LR={lr_stat:.3f}, df={df_diff:.0f}, p={p_val:.4g}"
        )

        # Interaction age * culture
        social_int = smf.logit(
            "social ~ age * C(culture) + majority_first", data=df
        ).fit(disp=False, maxiter=100)
        print("\nModel with age*culture interaction for social:")
        print(social_int.summary())

        lr_stat_int, df_diff_int, p_val_int = lr_test(social_culture, social_int)
        print(
            f"\nLR test for adding age*culture interaction to social model: "
            f"LR={lr_stat_int:.3f}, df={df_diff_int:.0f}, p={p_val_int:.4g}"
        )

        if "age" in social_base.pvalues:
            print(
                f"\nWald test for age in social base model: "
                f"coef={social_base.params['age']:.3f}, "
                f"p={social_base.pvalues['age']:.4g}"
            )
    except Exception as e:
        print("\n[Warning] Failed to fit social models:", repr(e))

    # Models for preference for majority cues (majority vs other choices)
    print("\n=== Logistic models: preference for majority cues (majority) ===")
    try:
        majority_base = smf.logit(
            "majority ~ age + majority_first", data=df
        ).fit(disp=False)
        print("\nBase model majority ~ age + majority_first")
        print(majority_base.summary())

        majority_culture = smf.logit(
            "majority ~ age + majority_first + C(culture)", data=df
        ).fit(disp=False)
        print(
            "\nModel with culture: majority ~ age + majority_first + C(culture)"
        )
        print(majority_culture.summary())

        lr_stat_m, df_diff_m, p_val_m = lr_test(majority_base, majority_culture)
        print(
            f"\nLR test for adding culture to majority model: "
            f"LR={lr_stat_m:.3f}, df={df_diff_m:.0f}, p={p_val_m:.4g}"
        )

        # Interaction age * culture
        majority_int = smf.logit(
            "majority ~ age * C(culture) + majority_first", data=df
        ).fit(disp=False, maxiter=100)
        print("\nModel with age*culture interaction for majority:")
        print(majority_int.summary())

        lr_stat_int_m, df_diff_int_m, p_val_int_m = lr_test(
            majority_culture, majority_int
        )
        print(
            f"\nLR test for adding age*culture interaction to majority model: "
            f"LR={lr_stat_int_m:.3f}, df={df_diff_int_m:.0f}, p={p_val_int_m:.4g}"
        )

        if "age" in majority_base.pvalues:
            print(
                f"\nWald test for age in majority base model: "
                f"coef={majority_base.params['age']:.3f}, "
                f"p={majority_base.pvalues['age']:.4g}"
            )
    except Exception as e:
        print("\n[Warning] Failed to fit majority models:", repr(e))


if __name__ == "__main__":
    main()

