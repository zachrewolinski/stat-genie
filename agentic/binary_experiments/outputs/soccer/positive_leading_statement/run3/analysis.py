import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv("soccer.csv")

    # Average the two raters to form a single skin tone measure
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)
    df = df[df["skin_mean"].notna()].copy()

    # Define light and dark groups using the extremes of the 5-point scale
    light = df[df["skin_mean"] <= 0.25]
    dark = df[df["skin_mean"] >= 0.75]

    def rate_per_game(g):
        total_red = g["redCards"].sum()
        total_games = g["games"].sum()
        return total_red / total_games if total_games > 0 else np.nan

    light_rate = rate_per_game(light)
    dark_rate = rate_per_game(dark)

    # Poisson regression on dyads with an exposure offset for games
    ld = df[(df["skin_mean"] <= 0.25) | (df["skin_mean"] >= 0.75)].copy()
    ld = ld[ld["games"] > 0]
    ld["is_dark"] = (ld["skin_mean"] >= 0.75).astype(int)

    X = sm.add_constant(ld["is_dark"])
    model = sm.GLM(
        ld["redCards"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(ld["games"]),
    )
    res = model.fit()

    # Print key results for inspection
    print("Light rate per game:", light_rate)
    print("Dark rate per game:", dark_rate)
    print("Poisson regression (dark vs light):")
    print(res.summary())


if __name__ == "__main__":
    main()
