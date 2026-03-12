import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Define key derived variables
    df["social"] = (df["y"] != 1).astype(int)
    df["majority"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    # Basic descriptives
    print("Overall sample size:", len(df))
    print("Overall reliance on social information (demonstrated options):", df["social"].mean())
    print(
        "Overall preference for majority among social choices:",
        df.loc[df["social"] == 1, "majority"].mean(),
    )

    # Descriptives by culture
    culture_summary = (
        df.groupby("culture")
        .agg(
            n=("y", "size"),
            social_rate=("social", "mean"),
            majority_rate=("majority", "mean"),
        )
        .sort_index()
    )
    print("\nSummary by culture:")
    print(culture_summary)

    # Descriptives by age quartiles (developmental stages proxy)
    df["age_group"] = pd.qcut(df["age"], 4, labels=["Q1_youngest", "Q2", "Q3", "Q4_oldest"])
    age_summary = (
        df.groupby("age_group")
        .agg(
            n=("y", "size"),
            mean_age=("age", "mean"),
            social_rate=("social", "mean"),
            majority_rate=("majority", "mean"),
        )
        .sort_values("mean_age")
    )
    print("\nSummary by age quartile:")
    print(age_summary)

    # Logistic regression: reliance on social information
    print("\nLogistic regression: reliance on social information (social ~ age + culture + gender + majority_first)")
    model_social = smf.logit(
        "social ~ age + C(culture) + gender + majority_first",
        data=df,
    ).fit(disp=False, maxiter=100)
    print(model_social.summary())

    # Logistic regression: preference for majority among social choices
    df_social = df[df["social"] == 1].copy()
    print(
        "\nLogistic regression: preference for majority (majority ~ age + culture + gender + majority_first) among social choosers"
    )
    model_majority = smf.logit(
        "majority ~ age + C(culture) + gender + majority_first",
        data=df_social,
    ).fit(disp=False, maxiter=100)
    print(model_majority.summary())


if __name__ == "__main__":
    main()

