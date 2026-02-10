import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise.
    df["focal_win"] = df["feature4"]

    # Relative group size (positive if focal group is larger).
    df["rel_group_size"] = df["feature7"] - df["feature8"]
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)

    # Contest location: which group is closer to the center of its home range.
    # Smaller distance means closer to home.
    df["focal_closer_home"] = (df["feature5"] < df["feature6"]).astype(int)

    print("Number of contests:", len(df))
    print()

    # Overall win rate.
    overall_win_rate = df["focal_win"].mean()
    print(f"Overall focal win rate: {overall_win_rate:.3f}")
    print()

    # Win rate by relative group size (focal larger vs not).
    print("Win rate by whether focal group is larger:")
    win_by_size = df.groupby("focal_larger")["focal_win"].agg(["mean", "count"])
    print(win_by_size)
    print()

    # Win rate by home-range proximity.
    print("Win rate by whether focal group is closer to home:")
    win_by_home = df.groupby("focal_closer_home")["focal_win"].agg(["mean", "count"])
    print(win_by_home)
    print()

    # Joint effects: relative size and location together.
    print("Win rate by size and home advantage (rows = size/location, cols = win outcome):")
    size_home_ct = pd.crosstab(
        [df["focal_larger"], df["focal_closer_home"]],
        df["focal_win"],
        normalize="index",
    )
    print(size_home_ct)
    print()

    # Simple linear probability model as a sanity check of directionality.
    X = df[["rel_group_size", "focal_closer_home"]].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])  # add intercept
    y = df["focal_win"].to_numpy(dtype=float)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef_names = ["intercept", "rel_group_size", "focal_closer_home"]
    print("Linear probability model coefficients (y = win probability):")
    for name, value in zip(coef_names, coef):
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()

