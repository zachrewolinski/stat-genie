import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    print("Head of data:")
    print(df.head())
    print("\nOutcome (y) value counts:")
    print(df["y"].value_counts().sort_index())

    # Social-information use: any demonstrated option (majority or minority) vs undemonstrated
    df["social"] = (df["y"] != 1).astype(int)

    print("\nProportion using social information overall:")
    print(df["social"].mean())

    print("\nProportion using social information by culture:")
    print(df.groupby("culture")["social"].mean())

    print("\nProportion choosing majority option (among social users) by culture:")
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)
    print(df_social.groupby("culture")["majority_choice"].mean())

    # Logistic regression: social-information use as a function of age and culture
    print("\nLogistic regression: social-information use ~ age + culture")
    model_social = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    print(model_social.summary())

    # Logistic regression: majority vs minority choice among those who used social information
    print("\nLogistic regression: majority_choice ~ age + culture (among social users)")
    model_majority = smf.logit("majority_choice ~ age + C(culture)", data=df_social).fit(
        disp=False
    )
    print(model_majority.summary())

    # Treat age as categorical developmental stages to allow non-linear effects
    print("\nLogistic regression: social-information use ~ C(age) + culture")
    model_social_cat = smf.logit("social ~ C(age) + C(culture)", data=df).fit(disp=False)
    print(model_social_cat.summary())

    print(
        "\nLogistic regression: majority_choice ~ C(age) + culture (among social users)"
    )
    model_majority_cat = smf.logit(
        "majority_choice ~ C(age) + C(culture)", data=df_social
    ).fit(disp=False)
    print(model_majority_cat.summary())


if __name__ == "__main__":
    main()
