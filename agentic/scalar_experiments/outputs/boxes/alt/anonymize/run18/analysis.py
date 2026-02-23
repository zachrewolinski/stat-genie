import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Recode outcomes
    df["social_use"] = df["feature1"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    df["site"] = df["feature5"].astype("category")

    # Logistic regression: reliance on social information ~ age + site
    model_social = smf.logit("social_use ~ feature3 + C(site)", data=df).fit(
        disp=False
    )

    # Logistic regression: majority preference among those who used social info
    df_social = df[df["social_use"] == 1].copy()
    model_majority = smf.logit(
        "majority_choice ~ feature3 + C(site)", data=df_social
    ).fit(disp=False)

    print("=== Reliance on social information (any demonstrated option) ===")
    print(model_social.summary())
    print("\n=== Preference for majority option among social users ===")
    print(model_majority.summary())


if __name__ == "__main__":
    main()

