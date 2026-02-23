import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"]

    # Relative group size: focal minus other
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Relative location: focal distance from its home-range center minus other's
    # Positive values mean focal is farther from its center than the other group.
    df["delta_distance"] = df["feature5"] - df["feature6"]

    X = df[["size_diff", "delta_distance"]]

    # Add intercept
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Number of contests:", len(df))
    print("\nLogistic regression of focal win (1) on:")
    print("  - size_diff = focal group size - other group size")
    print(
        "  - delta_distance = focal distance from home center - other distance "
        "from home center"
    )
    print("\nCoefficients (log-odds scale):")
    print(result.params)
    print("\nStandard errors:")
    print(result.bse)
    print("\nP-values:")
    print(result.pvalues)
    print("\nPseudo R-squared (McFadden):", result.prsquared)

    # Simple descriptive checks
    df["larger_focal"] = (df["size_diff"] > 0).astype(int)
    df["home_advantage_focal"] = (df["delta_distance"] < 0).astype(int)

    win_rate_larger = df.loc[df["larger_focal"] == 1, "feature4"].mean()
    win_rate_smaller = df.loc[df["larger_focal"] == 0, "feature4"].mean()

    win_rate_home = df.loc[df["home_advantage_focal"] == 1, "feature4"].mean()
    win_rate_away = df.loc[df["home_advantage_focal"] == 0, "feature4"].mean()

    print("\nWin rate when focal group is larger:", win_rate_larger)
    print("Win rate when focal group is not larger:", win_rate_smaller)
    print("\nWin rate when focal is closer to home center than other:", win_rate_home)
    print("Win rate otherwise:", win_rate_away)

    # Alternative specification with binary predictors for interpretability
    X_bin = df[["larger_focal", "home_advantage_focal"]]
    X_bin = sm.add_constant(X_bin)
    logit_bin = sm.Logit(y, X_bin).fit(disp=False)

    print(
        "\n\nLogistic regression with binary predictors:"
        "\n  - larger_focal = 1 if focal group has more individuals"
        "\n  - home_advantage_focal = 1 if focal is closer to its home center"
    )
    print("\nCoefficients (log-odds):")
    print(logit_bin.params)
    print("\nP-values:")
    print(logit_bin.pvalues)
    print("\nPseudo R-squared (McFadden):", logit_bin.prsquared)


if __name__ == "__main__":
    main()
