import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size (focal - other): positive if focal group is larger
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["focal_larger"] = (df["rel_size"] > 0).astype(int)

    # Relative location advantage: positive if focal is closer to its own home-range center
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]
    df["focal_closer"] = (df["rel_dist"] > 0).astype(int)

    y = df["win"]

    def run_logit(predictors):
        X = df[predictors]
        X = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y, X).fit(disp=False)
        return model

    # Logistic regression with continuous predictors
    model_cont = run_logit(["rel_size", "rel_dist"])

    # Logistic regression with binary advantage indicators
    model_bin = run_logit(["focal_larger", "focal_closer"])

    # Descriptive win rates
    size_win_rate = df.groupby("focal_larger")["win"].agg(["mean", "count"])
    loc_win_rate = df.groupby("focal_closer")["win"].agg(["mean", "count"])

    print("=== Logistic regression: continuous predictors (rel_size, rel_dist) ===")
    print("Coefficients:")
    print(model_cont.params)
    print("\nP-values:")
    print(model_cont.pvalues)
    print(f"\nMcFadden pseudo R^2: {model_cont.prsquared:.3f}")

    print("\n=== Logistic regression: binary predictors (focal_larger, focal_closer) ===")
    print("Coefficients:")
    print(model_bin.params)
    print("\nP-values:")
    print(model_bin.pvalues)
    print(f"\nMcFadden pseudo R^2: {model_bin.prsquared:.3f}")

    print("\n=== Descriptive win rates by size advantage (focal_larger) ===")
    print(size_win_rate)

    print("\n=== Descriptive win rates by location advantage (focal_closer) ===")
    print(loc_win_rate)


if __name__ == "__main__":
    main()

