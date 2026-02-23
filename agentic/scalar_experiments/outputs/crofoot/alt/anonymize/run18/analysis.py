import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise.
    y = df["feature4"]

    # Relative group size: focal group size minus other group size.
    rel_size = df["feature7"] - df["feature8"]

    # Relative location: other group's distance from its center minus focal group's.
    # Positive values mean the focal group is closer to the center of its home range
    # than the other group (a home-range advantage).
    loc_diff = df["feature6"] - df["feature5"]

    X = pd.DataFrame(
        {
            "rel_size": rel_size,
            "loc_diff": loc_diff,
        }
    )
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Logistic regression of focal win on relative size and location")
    print(result.summary())
    print("\nOdds ratios:")
    print(np.exp(result.params))
    print("\nP-values:")
    print(result.pvalues)

    # Simple descriptive checks
    df["rel_size"] = rel_size
    df["loc_diff"] = loc_diff
    print("\nMean rel_size (win=1 vs 0):")
    print(df.groupby("feature4")["rel_size"].mean())
    print("\nMean loc_diff (win=1 vs 0):")
    print(df.groupby("feature4")["loc_diff"].mean())


if __name__ == "__main__":
    main()

