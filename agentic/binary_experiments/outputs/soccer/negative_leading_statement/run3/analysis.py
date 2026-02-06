import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("soccer.csv")

    # Average the two raters' normalized skin tone scores (0=very light, 1=very dark)
    skin_mean = df[["rater1", "rater2"]].mean(axis=1)
    df = df.copy()
    df["skin_mean"] = skin_mean

    # Keep rows with skin ratings and positive exposure
    df = df[df["skin_mean"].notna() & (df["games"] > 0)].copy()

    # Define dark vs light: >= 0.5 treated as dark, < 0.5 as light
    df["dark"] = (df["skin_mean"] >= 0.5).astype(int)
    df["red_per_game"] = df["redCards"] / df["games"]

    group_stats = df.groupby("dark").agg(
        n=("redCards", "size"),
        mean_games=("games", "mean"),
        mean_red_cards=("redCards", "mean"),
        mean_red_per_game=("red_per_game", "mean"),
    )

    # Poisson regression with log(games) exposure offset
    X = sm.add_constant(df["dark"])
    model = sm.GLM(
        df["redCards"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(df["games"]),
    )
    res = model.fit(cov_type="HC0")

    rate_ratio = float(np.exp(res.params["dark"]))
    p_value = float(res.pvalues["dark"])

    print("Group stats (dark=0 light, dark=1 dark):")
    print(group_stats)
    print("\nPoisson GLM with exposure offset log(games):")
    print(res.summary())
    print(f"\nRate ratio (dark vs light): {rate_ratio:.4f}")
    print(f"P-value for dark indicator: {p_value:.4g}")


if __name__ == "__main__":
    main()
