import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables capturing social learning constructs
    df["social_choice"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )

    # Basic descriptives
    print("N observations:", len(df))
    print("\nOutcome distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())

    print("\nSocial choice rates by culture:")
    social_by_culture = (
        df.groupby("culture")["social_choice"].mean().rename("social_rate")
    )
    print(social_by_culture)

    print("\nMajority choice (among social choices) by culture:")
    maj_by_culture = (
        df[df["social_choice"] == 1]
        .groupby("culture")["majority_choice"]
        .mean()
        .rename("majority_rate")
    )
    print(maj_by_culture)

    print("\nSocial choice rate by age (mean and correlation):")
    print(
        df[["age", "social_choice"]]
        .groupby("age")
        .mean()
        .rename(columns={"social_choice": "mean_social"})
    )
    r_soc_age, p_soc_age = stats.pearsonr(df["age"], df["social_choice"])
    print(f"Correlation(age, social_choice): r={r_soc_age:.3f}, p={p_soc_age:.4f}")

    df_social = df[df["social_choice"] == 1].copy()
    r_maj_age, p_maj_age = stats.pearsonr(df_social["age"], df_social["majority_choice"])
    print(
        f"Correlation(age, majority_choice | social_choice): "
        f"r={r_maj_age:.3f}, p={p_maj_age:.4f}"
    )

    # Chi-square tests for culture effects
    print("\nChi-square test: social_choice by culture")
    ct_social_culture = pd.crosstab(df["culture"], df["social_choice"])
    chi2, p, dof, _ = stats.chi2_contingency(ct_social_culture)
    print("Chi2 =", chi2, "df =", dof, "p =", p)

    print("\nChi-square test: majority_choice by culture (among social choices)")
    ct_maj_culture = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    chi2_m, p_m, dof_m, _ = stats.chi2_contingency(ct_maj_culture)
    print("Chi2 =", chi2_m, "df =", dof_m, "p =", p_m)

    # Logistic regressions for age and culture effects
    print("\nLogistic regression: social_choice ~ age + C(culture) + gender + majority_first")
    model_social = smf.logit(
        "social_choice ~ age + C(culture) + gender + majority_first",
        data=df,
    ).fit(disp=False)
    print(model_social.summary())

    print(
        "\nLogistic regression: majority_choice ~ age + C(culture) + "
        "gender + majority_first (among social choices)"
    )
    model_majority = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        data=df_social,
    ).fit(disp=False)
    print(model_majority.summary())


if __name__ == "__main__":
    main()

