import numpy as np
import pandas as pd
import statsmodels.api as sm

def main():
    df = pd.read_csv("soccer.csv")

    # Skin tone: average of two raters (values 0..1 in 0.25 steps)
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)
    df = df.dropna(subset=["skin_mean", "games", "redCards", "playerShort"])

    # Aggregate to player level to avoid repeated dyads dominating
    player = (
        df.groupby("playerShort", as_index=False)
        .agg(
            skin_mean=("skin_mean", "mean"),
            games=("games", "sum"),
            redCards=("redCards", "sum"),
        )
    )

    # Categorize skin tone
    player["skin_cat"] = np.select(
        [player["skin_mean"] > 0.5, player["skin_mean"] < 0.5],
        ["dark", "light"],
        default="neutral",
    )

    pl = player[player["skin_cat"].isin(["dark", "light"])].copy()
    pl["red_rate"] = pl["redCards"] / pl["games"]

    # Descriptive stats
    summary = (
        pl.groupby("skin_cat")
        .agg(
            players=("playerShort", "count"),
            total_games=("games", "sum"),
            total_red=("redCards", "sum"),
            mean_rate=("red_rate", "mean"),
            overall_rate=("redCards", "sum"),
        )
    )
    summary["overall_rate"] = summary["overall_rate"] / summary["total_games"]

    print("Player-level summary (dark vs light):")
    print(summary)

    # Poisson regression on player-level counts with offset for exposure (games)
    X = pd.get_dummies(pl["skin_cat"], drop_first=True)  # dark=1, light=0
    X = sm.add_constant(X)
    model = sm.GLM(
        pl["redCards"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(pl["games"]),
    )
    res = model.fit()

    print("\nPoisson regression (redCards ~ dark, offset=log(games)):")
    print(res.summary())

    if "dark" in res.params.index:
        rate_ratio = float(np.exp(res.params["dark"]))
        p_value = float(res.pvalues["dark"])
        print(f"\nRate ratio (dark vs light): {rate_ratio:.3f}")
        print(f"P-value: {p_value:.4g}")

if __name__ == "__main__":
    main()
