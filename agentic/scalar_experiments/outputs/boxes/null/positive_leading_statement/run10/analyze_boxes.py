import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["rely_social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
    )

    print("=== Basic counts ===")
    print(f"N rows: {len(df)}")
    print(f"Overall reliance on social info (y in {{2,3}}): {df['rely_social'].mean():.3f}")
    follows = df["rely_social"] == 1
    print(
        "Overall majority preference among social followers (y==2 vs 3): "
        f"{df.loc[follows, 'majority_choice'].mean():.3f}"
    )

    print("\n=== Reliance on social information by culture ===")
    print(df.groupby("culture")["rely_social"].mean())

    print("\n=== Reliance on social information by age_group ===")
    print(df.groupby("age_group")["rely_social"].mean())

    print("\n=== Majority preference by culture (among social followers) ===")
    print(df.loc[follows].groupby("culture")["majority_choice"].mean())

    print("\n=== Majority preference by age_group (among social followers) ===")
    print(df.loc[follows].groupby("age_group")["majority_choice"].mean())

    print("\n=== Logistic regression: reliance on social information ===")
    model_rely = smf.logit("rely_social ~ age + C(culture)", data=df).fit(disp=False)
    print(model_rely.summary())
    print("\nP-values (rely_social model):")
    print(model_rely.pvalues)

    print("\n=== Logistic regression: majority preference among social followers ===")
    df_follow = df.loc[follows].copy()
    model_majority = smf.logit("majority_choice ~ age + C(culture)", data=df_follow).fit(
        disp=False
    )
    print(model_majority.summary())
    print("\nP-values (majority_choice model):")
    print(model_majority.pvalues)


if __name__ == "__main__":
    main()

