import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to semantic names based on info.json descriptions
    df = df.rename(
        columns={
            "m_focal": "win_focal",  # 1 if focal won, 0 otherwise
            "m_other": "focal_distance",  # distance of focal group to its home range center
            "n_focal": "other_distance",  # distance of other group to its home range center
            "f_other": "focal_size",  # number of individuals in focal group
            "win": "other_size",  # number of individuals in other group
        }
    )

    # Derived predictors: relative group size and relative location
    df["rel_group_size"] = df["focal_size"] - df["other_size"]
    df["rel_location"] = df["focal_distance"] - df["other_distance"]

    # Also create simple categorical indicators for descriptive summaries
    df["focal_larger"] = np.where(df["rel_group_size"] > 0, 1, 0)
    df["focal_smaller"] = np.where(df["rel_group_size"] < 0, 1, 0)
    df["focal_closer"] = np.where(df["rel_location"] < 0, 1, 0)
    df["other_closer"] = np.where(df["rel_location"] > 0, 1, 0)

    return df


def fit_logistic_models(df: pd.DataFrame):
    # Base design matrix with constant
    X = df[["rel_group_size", "rel_location"]].astype(float)
    X = sm.add_constant(X)
    y = df["win_focal"].astype(float)

    model = sm.Logit(y, X).fit(disp=False)
    return model


def summarize_results(df: pd.DataFrame, model: sm.Logit):
    print("=== Logistic regression: win_focal ~ rel_group_size + rel_location ===")
    print(model.summary())
    print("\nOdds ratios and 95% CIs:")
    params = model.params
    conf = model.conf_int()
    or_df = pd.DataFrame(
        {
            "odds_ratio": np.exp(params),
            "ci_lower": np.exp(conf[0]),
            "ci_upper": np.exp(conf[1]),
        }
    )
    print(or_df)

    # Simple descriptive summaries
    print("\n=== Descriptive summaries ===")
    print("Number of contests:", len(df))
    print(
        "Win rate when focal larger vs smaller/equal (by total group size):"
    )
    larger = df[df["rel_group_size"] > 0]
    not_larger = df[df["rel_group_size"] <= 0]
    print(
        "  Focal larger: n = {}, win rate = {:.2f}".format(
            len(larger), larger["win_focal"].mean()
        )
    )
    print(
        "  Focal not larger: n = {}, win rate = {:.2f}".format(
            len(not_larger), not_larger["win_focal"].mean()
        )
    )

    print(
        "\nWin rate when focal closer vs other closer to home range center:"
    )
    focal_closer = df[df["rel_location"] < 0]
    other_closer = df[df["rel_location"] > 0]
    neutral = df[df["rel_location"] == 0]
    if len(focal_closer) > 0:
        print(
            "  Focal closer: n = {}, win rate = {:.2f}".format(
                len(focal_closer), focal_closer["win_focal"].mean()
            )
        )
    if len(other_closer) > 0:
        print(
            "  Other closer: n = {}, win rate = {:.2f}".format(
                len(other_closer), other_closer["win_focal"].mean()
            )
        )
    if len(neutral) > 0:
        print(
            "  Equal distance: n = {}, win rate = {:.2f}".format(
                len(neutral), neutral["win_focal"].mean()
            )
        )


def main():
    csv_path = Path("crofoot.csv")
    df = load_data(csv_path)
    model = fit_logistic_models(df)
    summarize_results(df, model)

    # This script only prints analysis. The scalar response and narrative
    # explanation are written separately to conclusion.txt.


if __name__ == "__main__":
    main()

