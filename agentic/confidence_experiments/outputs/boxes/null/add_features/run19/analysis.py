import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Encode key outcomes related to the research question
    df["chose_demonstrated"] = df["y"].isin([2, 3]).astype(int)
    df["chose_majority"] = (df["y"] == 2).astype(int)

    # Treat culture as categorical; age is ordinal/continuous
    df["culture"] = df["culture"].astype("category")

    print("Overall outcome distribution (y):")
    print(df["y"].value_counts().sort_index(), "\n")

    print("Proportion choosing any demonstrated option by culture:")
    prop_demo_by_culture = df.groupby("culture")["chose_demonstrated"].mean()
    print(prop_demo_by_culture, "\n")

    print("Proportion choosing majority (among demonstrated choices) by culture:")
    mask_demo = df["chose_demonstrated"] == 1
    prop_maj_by_culture = (
        df.loc[mask_demo].groupby("culture")["chose_majority"].mean()
    )
    print(prop_maj_by_culture, "\n")

    print("Mean majority choice by age (raw age values):")
    print(
        df.groupby("age")["chose_majority"].mean(),
        "\n",
    )

    # Logistic regression: reliance on social information ~ age + culture
    model_social = smf.logit(
        "chose_demonstrated ~ age + C(culture)", data=df
    ).fit(disp=False)
    print("Logit model: chose_demonstrated ~ age + C(culture)")
    print(model_social.summary(), "\n")
    print("P-values (social information model):")
    print(model_social.pvalues, "\n")

    # Logistic regression: majority preference (conditioned on having chosen demonstrated)
    model_majority = smf.logit(
        "chose_majority ~ age + C(culture)", data=df.loc[mask_demo]
    ).fit(disp=False)
    print("Logit model: chose_majority ~ age + C(culture)")
    print(model_majority.summary(), "\n")
    print("P-values (majority preference model):")
    print(model_majority.pvalues, "\n")


if __name__ == "__main__":
    main()

