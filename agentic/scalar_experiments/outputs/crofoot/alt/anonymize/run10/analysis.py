import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["focal_won"] = df["feature4"].astype(int)

    # Relative group size: focal minus other, and size ratio
    df["size_diff"] = df["feature7"] - df["feature8"]
    df["size_ratio"] = df["feature7"] / df["feature8"]

    # Contest location advantage: how much closer the focal group is
    # to the center of its home range relative to the other group.
    # Positive values mean the focal group is closer to its own center.
    df["loc_advantage"] = df["feature6"] - df["feature5"]

    y = df["focal_won"]

    # Model 1: relative group size only (continuous difference)
    X_size = sm.add_constant(df[["size_diff"]])
    model_size = sm.GLM(y, X_size, family=sm.families.Binomial())
    res_size = model_size.fit()

    # Model 2: location advantage only (continuous)
    X_loc = sm.add_constant(df[["loc_advantage"]])
    model_loc = sm.GLM(y, X_loc, family=sm.families.Binomial())
    res_loc = model_loc.fit()

    # Model 3: both predictors together (continuous)
    X_both = sm.add_constant(df[["size_diff", "loc_advantage"]])
    model_both = sm.GLM(y, X_both, family=sm.families.Binomial())
    res_both = model_both.fit()

    print("=== Model with size_diff only ===")
    print(res_size.summary())
    print("\nOdds ratios:", np.exp(res_size.params))

    print("\n=== Model with loc_advantage only ===")
    print(res_loc.summary())
    print("\nOdds ratios:", np.exp(res_loc.params))

    print("\n=== Model with size_diff and loc_advantage ===")
    print(res_both.summary())
    print("\nOdds ratios:", np.exp(res_both.params))

    # Categorical versions for robustness checks
    df["focal_larger"] = np.where(df["size_diff"] > 0, 1, 0)
    df["focal_closer"] = np.where(df["loc_advantage"] > 0, 1, 0)

    print("\n=== Proportions: focal wins by focal_larger ===")
    print(df.groupby("focal_larger")["focal_won"].mean())

    print("\n=== Proportions: focal wins by focal_closer ===")
    print(df.groupby("focal_closer")["focal_won"].mean())

    # Logistic models with binary predictors
    X_bin_size = sm.add_constant(df[["focal_larger"]])
    res_bin_size = sm.GLM(y, X_bin_size, family=sm.families.Binomial()).fit()
    print("\n=== Logistic model with focal_larger (binary) ===")
    print(res_bin_size.summary())
    print("\nOdds ratios:", np.exp(res_bin_size.params))

    X_bin_loc = sm.add_constant(df[["focal_closer"]])
    res_bin_loc = sm.GLM(y, X_bin_loc, family=sm.families.Binomial()).fit()
    print("\n=== Logistic model with focal_closer (binary) ===")
    print(res_bin_loc.summary())
    print("\nOdds ratios:", np.exp(res_bin_loc.params))


if __name__ == "__main__":
    main()
