import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic recodes
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["any_demonstrated"] = df["y"].isin([2, 3]).astype(int)
    df["majority_vs_minority"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    print("N total:", len(df))
    print("Majority choice rate overall:", df["majority_choice"].mean())
    print("Any demonstrated option chosen rate:", df["any_demonstrated"].mean())

    # Group summaries by culture
    culture_summary = (
        df.groupby("culture")[["majority_choice", "any_demonstrated"]]
        .mean()
        .rename(columns={"majority_choice": "p_majority", "any_demonstrated": "p_any_demo"})
    )
    print("\nBy culture (proportions):")
    print(culture_summary)

    # Group summaries by age quartiles to approximate developmental stages
    df["age_bin"] = pd.qcut(df["age"], q=4, duplicates="drop")
    age_summary = (
        df.groupby("age_bin")[["majority_choice", "any_demonstrated"]]
        .mean()
        .rename(columns={"majority_choice": "p_majority", "any_demonstrated": "p_any_demo"})
    )
    print("\nBy age quartile (proportions):")
    print(age_summary)

    # Logistic regression: majority vs others as function of age and culture
    logit1 = smf.logit("majority_choice ~ age + C(culture)", data=df).fit(disp=False)
    print("\nLogit: majority_choice ~ age + C(culture)")
    print(logit1.summary())

    # Logistic regression: majority vs minority among demonstrated choices only
    df_demo = df[df["y"].isin([2, 3])].copy()
    logit2 = smf.logit("majority_vs_minority ~ age + C(culture)", data=df_demo).fit(disp=False)
    print("\nLogit among demonstrated choices: majority_vs_minority ~ age + C(culture)")
    print(logit2.summary())


if __name__ == "__main__":
    main()

