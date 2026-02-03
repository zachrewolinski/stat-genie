import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Skin tone mean from two raters (0=very light, 1=very dark)
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

    # Define clear light vs dark groups; exclude the neutral midpoint (0.5)
    df = df[df["skin_mean"].notna()].copy()
    df["skin_group"] = np.where(df["skin_mean"] > 0.5, "dark",
                                 np.where(df["skin_mean"] < 0.5, "light", "mid"))
    df = df[df["skin_group"].isin(["dark", "light"])].copy()

    # Aggregate to player level to avoid dyad duplication of skin tone
    agg = df.groupby("playerShort", as_index=False).agg(
        skin_mean=("skin_mean", "mean"),
        skin_group=("skin_group", "first"),
        total_games=("games", "sum"),
        total_red=("redCards", "sum"),
    )

    # Player-level red card rate per game
    agg["red_rate"] = agg["total_red"] / agg["total_games"].replace(0, np.nan)

    # Summary by group
    group_summary = agg.groupby("skin_group").agg(
        players=("playerShort", "nunique"),
        total_games=("total_games", "sum"),
        total_red=("total_red", "sum"),
        mean_player_rate=("red_rate", "mean"),
    )
    group_summary["rate_per_game"] = group_summary["total_red"] / group_summary["total_games"]

    # Poisson regression: total_red ~ dark, offset log(total_games)
    agg = agg[agg["total_games"] > 0].copy()
    agg["dark"] = (agg["skin_group"] == "dark").astype(int)
    X = sm.add_constant(agg["dark"])
    y = agg["total_red"]
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(agg["total_games"]))
    res = model.fit()

    print("Group summary (player-level):")
    print(group_summary)
    print("\nPoisson regression (offset log games):")
    print(res.summary())


if __name__ == "__main__":
    main()
