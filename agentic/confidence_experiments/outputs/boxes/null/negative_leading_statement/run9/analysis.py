import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Indicators for reliance on social information and majority preference
    df["social_choice"] = (df["y"] != 1).astype(int)  # chose any demonstrated option
    df["majority_choice"] = (df["y"] == 2).astype(int)  # chose majority option

    print("N =", len(df))
    print("\nProportion choosing any demonstrated option (social_choice=1) by culture:")
    print(df.groupby("culture")["social_choice"].mean())

    print("\nProportion choosing majority option (majority_choice=1) by culture:")
    print(df.groupby("culture")["majority_choice"].mean())

    # Age as continuous predictor; culture as categorical; control for gender and order
    print("\nLogistic regression: social_choice ~ age + C(culture) + gender + majority_first")
    model_social = smf.logit(
        "social_choice ~ age + C(culture) + gender + majority_first",
        data=df,
    ).fit(disp=False)
    print(model_social.summary())

    print("\nLogistic regression: majority_choice ~ age + C(culture) + gender + majority_first")
    model_majority = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        data=df,
    ).fit(disp=False)
    print(model_majority.summary())

    # Also inspect age trends directly
    print("\nProportion choosing majority option by age:")
    print(df.groupby("age")["majority_choice"].mean())


if __name__ == "__main__":
    main()

