import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # According to info.json descriptions:
    # - m_focal: 1 if focal won contest, 0 otherwise
    # - f_other: number of individuals in focal group
    # - win: number of individuals in other group
    # - m_other: distance (m) of focal group from center of its home range
    # - n_focal: distance (m) of other group from center of its home range
    #
    # Derive interpretable predictors for relative group size and contest location.

    # Group size features
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]
    df["size_diff"] = df["size_focal"] - df["size_other"]
    df["size_ratio"] = df["size_focal"] / df["size_other"]
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)

    # Contest location features (home-range distances)
    df["dist_home_focal"] = df["m_other"]
    df["dist_home_other"] = df["n_focal"]
    df["dist_diff_home"] = df["dist_home_focal"] - df["dist_home_other"]
    df["focal_home_adv"] = (df["dist_home_focal"] < df["dist_home_other"]).astype(int)

    return df


def summarise_effects(df: pd.DataFrame) -> None:
    print("Dataset size:", len(df))
    print("\nOverall focal win rate:", df["m_focal"].mean())

    # Simple descriptive relationships
    print("\nWin rate by relative group size (focal larger vs not):")
    print(
        df.groupby("focal_larger")["m_focal"]
        .agg(["mean", "count"])
        .rename(index={0: "focal_not_larger", 1: "focal_larger"})
    )

    print("\nWin rate by home-range advantage (focal closer to its center):")
    print(
        df.groupby("focal_home_adv")["m_focal"]
        .agg(["mean", "count"])
        .rename(index={0: "no_home_adv", 1: "focal_home_adv"})
    )

    # Logistic regression for a more formal check
    features = df[["size_diff", "dist_diff_home", "focal_larger", "focal_home_adv"]]
    features = sm.add_constant(features)
    model = sm.Logit(df["m_focal"], features)
    result = model.fit(disp=False)

    print("\nLogistic regression results:")
    print(result.summary())


def main() -> None:
    data_path = Path("crofoot.csv")
    df = load_data(str(data_path))
    summarise_effects(df)


if __name__ == "__main__":
    main()

