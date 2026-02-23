import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic sanity checks
    print("Head:")
    print(df.head())
    print("\nValue counts for y (1=unchosen, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())

    # Create derived outcomes
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("\nProportion using social information (any demonstrator):")
    print(df["social_choice"].mean())
    print("Proportion choosing majority option (overall):")
    print(df["majority_choice"].mean())

    # Treat culture as categorical factor
    df["culture"] = df["culture"].astype("category")

    # Center age for interpretability
    df["age_c"] = df["age"] - df["age"].mean()

    # Logistic regression: reliance on social information
    print("\n=== Logistic regression: social_choice ~ age_c + culture + majority_first ===")
    model_social = smf.logit(
        "social_choice ~ age_c + C(culture) + majority_first", data=df
    ).fit(disp=0)
    print(model_social.summary())

    # Logistic regression: majority vs minority, among social choosers
    df_social = df[df["social_choice"] == 1].copy()
    print("\nNumber of social choosers:", len(df_social))
    print("\n=== Logistic regression: majority_choice ~ age_c + culture + majority_first (among social choosers) ===")
    model_majority = smf.logit(
        "majority_choice ~ age_c + C(culture) + majority_first", data=df_social
    ).fit(disp=0)
    print(model_majority.summary())

    # Marginal probabilities by culture (descriptive)
    print("\nMean majority choice by culture:")
    print(df.groupby("culture")["majority_choice"].mean())

    print("\nMean social choice by culture:")
    print(df.groupby("culture")["social_choice"].mean())


if __name__ == "__main__":
    main()

