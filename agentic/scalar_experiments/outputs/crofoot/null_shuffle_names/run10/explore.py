import pandas as pd


def main() -> None:
    df = pd.read_csv("crofoot.csv")
    focal_size = df["f_other"]
    other_size = df["win"]
    focal_dist = df["m_other"]
    other_dist = df["n_focal"]

    df["size_diff"] = focal_size - other_size
    df["dist_diff"] = other_dist - focal_dist

    print("Overall win rate:", df["m_focal"].mean())

    for label, cond in [
        ("focal larger", df["size_diff"] > 0),
        ("equal size", df["size_diff"] == 0),
        ("focal smaller", df["size_diff"] < 0),
    ]:
        sub = df[cond]
        print(label, "n=", len(sub), "win rate=", sub["m_focal"].mean())

    for label, cond in [
        ("closer to focal", df["dist_diff"] > 0),
        ("equal distance", df["dist_diff"] == 0),
        ("closer to other", df["dist_diff"] < 0),
    ]:
        sub = df[cond]
        print(label, "n=", len(sub), "win rate=", sub["m_focal"].mean())


if __name__ == "__main__":
    main()

