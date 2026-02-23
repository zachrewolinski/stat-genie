import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Derived predictors
    df["size_diff"] = df["feature7"] - df["feature8"]
    df["size_ratio"] = df["feature7"] / df["feature8"]
    df["loc_diff"] = df["feature5"] - df["feature6"]  # focal distance minus other distance
    df["focal_closer_home"] = (df["feature5"] < df["feature6"]).astype(int)

    print("Basic description of key variables")
    print(df[["feature4", "feature7", "feature8", "size_diff", "loc_diff"]].describe())
    print()

    # Logistic regression: win (feature4) on relative group size and location
    model = smf.logit("feature4 ~ size_diff + loc_diff", data=df)
    result = model.fit(disp=False)

    print("Logistic regression: win ~ size_diff + loc_diff")
    print(result.summary())
    print()

    # Also check a model with size_ratio instead of size_diff
    model_ratio = smf.logit("feature4 ~ size_ratio + loc_diff", data=df)
    result_ratio = model_ratio.fit(disp=False)

    print("Logistic regression: win ~ size_ratio + loc_diff")
    print(result_ratio.summary())
    print()

    # Simpler models to probe individual effects
    model_size_only = smf.logit("feature4 ~ size_diff", data=df)
    result_size_only = model_size_only.fit(disp=False)

    print("Logistic regression: win ~ size_diff")
    print(result_size_only.summary())
    print()

    model_loc_only = smf.logit("feature4 ~ loc_diff", data=df)
    result_loc_only = model_loc_only.fit(disp=False)

    print("Logistic regression: win ~ loc_diff")
    print(result_loc_only.summary())
    print()

    # Model using a more interpretable binary home-field indicator
    model_home = smf.logit("feature4 ~ focal_closer_home + size_diff", data=df)
    result_home = model_home.fit(disp=False)

    print("Logistic regression: win ~ focal_closer_home + size_diff")
    print(result_home.summary())


if __name__ == "__main__":
    main()
