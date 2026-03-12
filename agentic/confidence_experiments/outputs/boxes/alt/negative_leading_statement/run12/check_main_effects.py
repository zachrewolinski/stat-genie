import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    print("=== Social choice main-effects model ===")
    model_social_main = smf.glm(
        "social ~ age + C(culture) + gender + majority_first",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social_main.pvalues)

    print("\n=== Majority vs minority main-effects model ===")
    model_majority_main = smf.glm(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority_main.pvalues)


if __name__ == "__main__":
    main()

