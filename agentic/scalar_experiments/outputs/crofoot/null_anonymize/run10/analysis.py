import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won.
    y = df["feature4"]

    # Relative group size: focal minus other.
    rel_group_size = df["feature7"] - df["feature8"]

    # Contest location: relative distance to home range centre (other minus focal).
    rel_distance_centre = df["feature6"] - df["feature5"]

    X = pd.DataFrame(
        {
            "rel_group_size": rel_group_size,
            "rel_distance_centre": rel_distance_centre,
        }
    )
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    print("Logit coefficients:")
    print(model.params)
    print("\nP-values:")
    print(model.pvalues)
    print("\nPseudo R-squared (McFadden):", model.prsquared)

    # Descriptive win rates by relative group size category.
    df["rel_group_size"] = rel_group_size
    df["rel_distance_centre"] = rel_distance_centre

    def win_rate(sub):
        return sub["feature4"].mean()

    print("\nWin rate by relative group size category:")
    size_cat = pd.cut(
        df["rel_group_size"],
        bins=[-100, -1, 0, 1, 100],
        labels=["focal smaller", "equal", "focal slightly larger", "focal much larger"],
    )
    print(df.groupby(size_cat).apply(win_rate))

    print("\nWin rate by contest location (closer to home):")
    location_cat = pd.cut(
        df["rel_distance_centre"],
        bins=[-10_000, -1, 0, 1, 10_000],
        labels=["closer to focal", "equal", "closer to other", "much closer to other"],
    )
    print(df.groupby(location_cat).apply(win_rate))


if __name__ == "__main__":
    main()
