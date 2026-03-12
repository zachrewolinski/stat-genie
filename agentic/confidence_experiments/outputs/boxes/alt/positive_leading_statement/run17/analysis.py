import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social information reliance: 1 if child follows any demonstrator (majority or minority)
    df["social"] = (df["y"] != 1).astype(int)

    # Majority preference among social learners: 1 = majority, 0 = minority (NaN if no social choice)
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    # Coarse developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 14],
        labels=["4-6", "7-9", "10-14"],
        include_lowest=True,
    )

    print("N observations:", len(df))
    print("Overall social reliance rate (y != 1):", df["social"].mean())
    social_learners = df[df["social"] == 1]
    print(
        "Overall majority preference among social learners (P(majority | social)):",
        social_learners["majority_choice"].mean(),
    )

    # Descriptive by culture and age group
    desc = (
        df.groupby(["culture", "age_group"])
        .agg(
            n=("y", "size"),
            social_rate=("social", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )
    print("\nDescriptive rates by culture and age_group:")
    print(desc.to_string(index=False))

    # Logistic regression: social reliance ~ age + culture + controls
    print("\n--- Logistic model: social reliance ~ age + culture + gender + majority_first ---")
    model_social = smf.logit(
        "social ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=False)
    print(model_social.summary())
    model_social_base = smf.logit(
        "social ~ age + gender + majority_first", data=df
    ).fit(disp=False)
    lr_stat = 2 * (model_social.llf - model_social_base.llf)
    df_diff = model_social.df_model - model_social_base.df_model
    p_lr = chi2.sf(lr_stat, df_diff)
    print(
        f"Likelihood-ratio test for culture in social model: chi2({df_diff:.0f})={lr_stat:.3f}, p={p_lr:.5f}"
    )

    # Logistic regression: majority preference among social learners
    print(
        "\n--- Logistic model: majority preference (among social learners) ~ age + culture + gender + majority_first ---"
    )
    model_maj = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        data=social_learners,
    ).fit(disp=False)
    print(model_maj.summary())
    model_maj_base = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=social_learners
    ).fit(disp=False)
    lr_stat2 = 2 * (model_maj.llf - model_maj_base.llf)
    df_diff2 = model_maj.df_model - model_maj_base.df_model
    p_lr2 = chi2.sf(lr_stat2, df_diff2)
    print(
        f"Likelihood-ratio test for culture in majority model: chi2({df_diff2:.0f})={lr_stat2:.3f}, p={p_lr2:.5f}"
    )

    # Models with developmental stages (age_group) instead of continuous age
    df_stage = df.dropna(subset=["age_group"]).copy()
    print(
        "\n--- Logistic model: social reliance ~ age_group + culture + gender + majority_first ---"
    )
    model_social_stage = smf.logit(
        "social ~ C(age_group) + C(culture) + gender + majority_first", data=df_stage
    ).fit(disp=False)
    print(model_social_stage.summary())

    social_learners_stage = df_stage[df_stage["social"] == 1].copy()
    print(
        "\n--- Logistic model: majority preference ~ age_group + culture + gender + majority_first (social learners only) ---"
    )
    model_maj_stage = smf.logit(
        "majority_choice ~ C(age_group) + C(culture) + gender + majority_first",
        data=social_learners_stage,
    ).fit(disp=False)
    print(model_maj_stage.summary())


if __name__ == "__main__":
    main()

