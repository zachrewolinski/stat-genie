import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Sanity checks
    print("Shape:", df.shape)
    print(df.head())

    # Outcome recodes
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["social_reliance"] = df["y"].isin([2, 3]).astype(int)

    # Treat culture as categorical and age as continuous
    df["culture"] = df["culture"].astype("category")

    # Also define coarse age groups (tertiles) as a proxy for developmental stages
    df["age_group"] = pd.qcut(df["age"], 3, labels=["young", "middle", "older"])

    print("\nOverall outcome distribution (y):")
    print(df["y"].value_counts(normalize=True))

    print("\nMajority choice rate by culture:")
    print(
        df.groupby("culture")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .sort_values()
    )

    print("\nMajority choice rate by age_group:")
    print(
        df.groupby("age_group")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .sort_values()
    )

    print("\nSocial reliance rate (y in {2,3}) by culture:")
    print(
        df.groupby("culture")["social_reliance"]
        .mean()
        .rename("social_rate")
        .sort_values()
    )

    print("\nSocial reliance rate by age_group:")
    print(
        df.groupby("age_group")["social_reliance"]
        .mean()
        .rename("social_rate")
        .sort_values()
    )

    # Logistic regression: majority choice ~ age + culture
    print("\nLogistic regression: majority_choice ~ age + culture")
    logit_mod = smf.logit("majority_choice ~ age + C(culture)", data=df)
    logit_res = logit_mod.fit(disp=False)
    print(logit_res.summary())

    # Logistic regression: social reliance ~ age + culture
    print("\nLogistic regression: social_reliance ~ age + culture")
    logit_soc_mod = smf.logit("social_reliance ~ age + C(culture)", data=df)
    logit_soc_res = logit_soc_mod.fit(disp=False)
    print(logit_soc_res.summary())


if __name__ == "__main__":
    main()

