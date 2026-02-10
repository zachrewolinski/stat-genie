import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    win = df["m_focal"].astype(int)

    # Group sizes (total individuals)
    size_focal = df["f_other"]
    size_other = df["win"]

    # Distances from each group's home-range center (meters)
    dist_focal = df["m_other"]
    dist_other = df["n_focal"]

    analysis = pd.DataFrame(
        {
            "win": win,
            "size_focal": size_focal,
            "size_other": size_other,
            "dist_focal": dist_focal,
            "dist_other": dist_other,
        }
    )

    # Relative group size categories
    analysis["size_cat"] = np.where(
        analysis["size_focal"] > analysis["size_other"],
        "focal_larger",
        np.where(
            analysis["size_focal"] < analysis["size_other"],
            "focal_smaller",
            "equal",
        ),
    )

    size_summary = (
        analysis.groupby("size_cat")["win"].agg(["mean", "count"]).to_dict("index")
    )

    # Relative location (who is closer to own home-range center)
    # Positive rel_dist => focal is closer; negative => other is closer
    analysis["rel_dist"] = analysis["dist_other"] - analysis["dist_focal"]
    analysis["loc_cat"] = np.where(
        analysis["rel_dist"] > 0,
        "focal_closer",
        np.where(analysis["rel_dist"] < 0, "other_closer", "equal"),
    )

    loc_summary = (
        analysis.groupby("loc_cat")["win"].agg(["mean", "count"]).to_dict("index")
    )

    # Effect of relative group size on win probability
    p_focal_larger = size_summary.get("focal_larger", {}).get("mean", 0.0)
    p_focal_smaller = size_summary.get("focal_smaller", {}).get("mean", 0.0)
    size_diff = p_focal_larger - p_focal_smaller

    # Effect of contest location on win probability
    p_focal_closer = loc_summary.get("focal_closer", {}).get("mean", 0.0)
    p_other_closer = loc_summary.get("other_closer", {}).get("mean", 0.0)
    loc_diff = p_focal_closer - p_other_closer

    # Combine absolute effects from size and location into a single Likert-style scalar.
    # Each difference is in [-1, 1]; their average is in [-1, 1] as well.
    effect_strength = 0.5 * (abs(size_diff) + abs(loc_diff))
    effect_strength = max(0.0, min(1.0, effect_strength))

    scalar = int(round(effect_strength * 100))

    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

