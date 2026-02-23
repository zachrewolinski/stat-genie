import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Construct relative group size and relative location predictors
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["rel_location"] = df["dist_other"] - df["dist_focal"]

    # Drop any rows with missing values in the variables of interest (if any)
    df_model = df.dropna(subset=["win", "rel_group_size", "rel_location"])

    y = df_model["win"]

    # Model 1: main effects only
    X1 = df_model[["rel_group_size", "rel_location"]]
    X1 = sm.add_constant(X1)

    logit_model1 = sm.Logit(y, X1)
    result1 = logit_model1.fit(disp=False)

    print("Model 1: win ~ rel_group_size + rel_location")
    print(result1.summary())

    params1 = result1.params
    conf1 = result1.conf_int()
    odds_ratios1 = np.exp(params1)
    conf_odds1 = np.exp(conf1)

    print("\nModel 1 odds ratios with 95% CI:")
    for name in params1.index:
        or_val = odds_ratios1[name]
        ci_low, ci_high = conf_odds1.loc[name]
        print(f"{name:15s} OR={or_val:6.3f}  95% CI=({ci_low:6.3f}, {ci_high:6.3f})")

    # Model 2: include interaction between relative group size and location
    df_model["rel_interaction"] = df_model["rel_group_size"] * df_model["rel_location"]
    X2 = df_model[["rel_group_size", "rel_location", "rel_interaction"]]
    X2 = sm.add_constant(X2)

    logit_model2 = sm.Logit(y, X2)
    result2 = logit_model2.fit(disp=False)

    print("\nModel 2: win ~ rel_group_size * rel_location")
    print(result2.summary())

    params2 = result2.params
    conf2 = result2.conf_int()
    odds_ratios2 = np.exp(params2)
    conf_odds2 = np.exp(conf2)

    print("\nModel 2 odds ratios with 95% CI:")
    for name in params2.index:
        or_val = odds_ratios2[name]
        ci_low, ci_high = conf_odds2.loc[name]
        print(f"{name:15s} OR={or_val:6.3f}  95% CI=({ci_low:6.3f}, {ci_high:6.3f})")


if __name__ == "__main__":
    main()
