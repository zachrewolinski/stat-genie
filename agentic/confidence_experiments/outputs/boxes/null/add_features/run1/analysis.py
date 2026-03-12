import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic sanity checks
    print("Columns:", list(df.columns))
    print("N rows:", len(df))
    print("\nOutcome distribution (y):")
    print(df["y"].value_counts().sort_index())

    # Derived variables
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    print("\nSocial choice rate (any culture):", df["social"].mean())
    print("Majority among social choosers:", df.loc[df["y"].isin([2, 3]), "majority_choice"].mean())

    # Treat culture as categorical
    df["culture"] = df["culture"].astype("category")

    # SOCIAL VS ASOCIAL MODEL
    print("\n=== Logistic regression: social (2/3) vs asocial (1) ===")
    social_full = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    social_age_only = smf.logit("social ~ age", data=df).fit(disp=False)
    social_culture_only = smf.logit("social ~ C(culture)", data=df).fit(disp=False)

    # Age effect (Wald test)
    p_age_social = float(social_full.pvalues["age"])

    # Culture effect (likelihood-ratio test vs age-only model)
    lr_culture_social = 2 * (social_full.llf - social_age_only.llf)
    df_culture_social = social_full.df_model - social_age_only.df_model
    p_culture_social = float(1 - stats.chi2.cdf(lr_culture_social, df_culture_social))

    print("Social model coefficients:")
    print(social_full.params)
    print("P-value for age (social):", p_age_social)
    print("LR test for culture given age (social): stat=", lr_culture_social, "df=", df_culture_social, "p=", p_culture_social)

    # MAJORITY VS MINORITY AMONG SOCIAL CHOOSERS
    print("\n=== Logistic regression: majority (2) vs minority (3) choices among social choosers ===")
    df_social = df[df["y"].isin([2, 3])].copy()
    majority_full = smf.logit("majority_choice ~ age + C(culture)", data=df_social).fit(disp=False)
    majority_age_only = smf.logit("majority_choice ~ age", data=df_social).fit(disp=False)

    # Age effect (Wald test)
    p_age_majority = float(majority_full.pvalues["age"])

    # Culture effect (LR test vs age-only model)
    lr_culture_majority = 2 * (majority_full.llf - majority_age_only.llf)
    df_culture_majority = majority_full.df_model - majority_age_only.df_model
    p_culture_majority = float(1 - stats.chi2.cdf(lr_culture_majority, df_culture_majority))

    print("Majority model coefficients:")
    print(majority_full.params)
    print("P-value for age (majority vs minority):", p_age_majority)
    print(
        "LR test for culture given age (majority vs minority): "
        f"stat={lr_culture_majority}, df={df_culture_majority}, p={p_culture_majority}"
    )

    # Simple descriptive summaries by culture and age quartile
    df["age_q"] = pd.qcut(df["age"], q=4, duplicates="drop")
    print("\nSocial choice rate by culture:")
    print(df.groupby("culture")["social"].mean())
    print("\nMajority choice among social choosers by culture:")
    print(df_social.groupby("culture")["majority_choice"].mean())

    print("\nSocial choice rate by age quartile:")
    print(df.groupby("age_q")["social"].mean())
    print("\nMajority choice among social choosers by age quartile:")
    print(
        df_social.groupby(pd.qcut(df_social["age"], q=4, duplicates="drop"))["majority_choice"].mean()
    )


if __name__ == "__main__":
    main()
