import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Map columns to meaningful names for clarity
    df = df.rename(
        columns={
            "feature1": "outcome",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Basic derived measures
    df["is_social"] = df["outcome"].isin([2, 3]).astype(int)
    df["is_majority"] = (df["outcome"] == 2).astype(int)
    df["age_c"] = df["age"] - df["age"].mean()

    # Logistic regression: reliance on social information (social vs non-social)
    # Predictors: age (centered), site (categorical)
    logit_social = smf.logit(
        "is_social ~ age_c + C(site)", data=df
    ).fit(disp=False)

    # Logistic regression: majority vs minority, only among social choices
    social_df = df[df["is_social"] == 1].copy()
    logit_majority = smf.logit(
        "is_majority ~ age_c + C(site)", data=social_df
    ).fit(disp=False)

    # Summaries for interpretation (printed to stdout for human inspection)
    print("=== Reliance on social information (social vs non-social) ===")
    print(logit_social.summary())
    print()

    print("=== Majority preference among social choices (majority vs minority) ===")
    print(logit_majority.summary())
    print()

    # Simple descriptive variation across sites and ages
    site_social_rates = (
        df.groupby("site")["is_social"].mean().describe()[["min", "max", "std"]]
    )
    site_majority_rates = (
        social_df.groupby("site")["is_majority"].mean().describe()[["min", "max", "std"]]
    )

    print("Social-choice rate across sites (min, max, std):")
    print(site_social_rates)
    print()
    print("Majority-choice rate across sites (min, max, std):")
    print(site_majority_rates)
    print()

    age_bins = pd.cut(df["age"], bins=[4, 6, 8, 10, 12, 14], include_lowest=True)
    age_social_rates = df.groupby(age_bins)["is_social"].mean()
    age_majority_rates = social_df.groupby(age_bins)["is_majority"].mean()
    print("Social-choice rate by age bin:")
    print(age_social_rates)
    print()
    print("Majority-choice rate by age bin:")
    print(age_majority_rates)


if __name__ == "__main__":
    main()

