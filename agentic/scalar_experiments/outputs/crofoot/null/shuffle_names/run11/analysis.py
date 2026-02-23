import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Rename outcome for clarity
    df["focal_win"] = df["m_focal"]

    # Relative group size: focal minus other
    df["rel_group_size"] = df["f_other"] - df["win"]
    df["rel_group_ratio"] = df["f_other"] / df["win"]

    # Contest location: distance from each group's home-range center
    df["rel_home_distance"] = df["m_other"] - df["n_focal"]
    df["focal_closer_home"] = (df["m_other"] <= df["n_focal"]).astype(int)

    print("Shape:", df.shape)
    print("\nWin counts (focal_win):")
    print(df["focal_win"].value_counts())

    print("\nRelative group size summary:")
    print(df["rel_group_size"].describe())

    print("\nRelative home distance summary (m_other - n_focal):")
    print(df["rel_home_distance"].describe())

    # Empirical win rates by relative group size category
    def size_category(x: int) -> str:
        if x <= -2:
            return "focal much smaller (<= -2)"
        if x == -1:
            return "focal slightly smaller (-1)"
        if x == 0:
            return "same size (0)"
        if x == 1:
            return "focal slightly larger (1)"
        return "focal much larger (>= 2)"

    df["size_cat"] = df["rel_group_size"].apply(size_category)
    print("\nWin rate by relative group size category:")
    print(
        df.groupby("size_cat")["focal_win"]
        .agg(["mean", "count"])
        .sort_index()
    )

    # Empirical win rates by home-advantage indicator
    print("\nWin rate by focal_closer_home (1=focal closer to home center):")
    print(df.groupby("focal_closer_home")["focal_win"].agg(["mean", "count"]))

    # Logistic regression models
    def fit_and_print(formula: str, name: str) -> None:
        print(f"\n=== {name} ===")
        try:
            model = smf.logit(formula, data=df).fit(disp=False)
            print(model.summary())
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"Failed to fit {name}: {exc}")

    fit_and_print("focal_win ~ rel_group_size", "Model 1: size only")
    fit_and_print("focal_win ~ rel_home_distance", "Model 2: location only")
    fit_and_print(
        "focal_win ~ rel_group_size + rel_home_distance",
        "Model 3: size + location",
    )
    fit_and_print(
        "focal_win ~ rel_group_size + focal_closer_home",
        "Model 4: size + home-advantage indicator",
    )


if __name__ == "__main__":
    main()

