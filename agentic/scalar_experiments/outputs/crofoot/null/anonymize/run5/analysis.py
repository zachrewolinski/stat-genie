import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"]

    # Relative group size: focal group size minus other group size
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Location advantage: how much closer the focal group is to the center of
    # its own home range compared with the other group.
    # Positive values mean the focal group is more "at home".
    df["loc_advantage"] = df["feature6"] - df["feature5"]

    X = df[["rel_group_size", "loc_advantage"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression: focal win ~ rel_group_size + loc_advantage")
    print(result.summary())

    # Also fit single-predictor models for robustness
    for col in ["rel_group_size", "loc_advantage"]:
        Xi = sm.add_constant(df[[col]])
        model_i = sm.Logit(y, Xi).fit(disp=False)
        print(f"\nLogistic regression with single predictor: {col}")
        print(model_i.summary())


if __name__ == "__main__":
    main()

