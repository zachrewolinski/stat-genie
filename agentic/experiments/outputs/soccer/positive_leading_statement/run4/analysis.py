import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv("soccer.csv")

    # Skin tone: average of two raters on 0-1 scale (5-point).
    df["skin"] = df[["rater1", "rater2"]].mean(axis=1)

    # Define light vs dark; exclude middle (0.5) to keep a clean binary split.
    df = df[df["skin"].notna() & (df["games"] > 0)]
    df = df[(df["skin"] < 0.5) | (df["skin"] > 0.5)].copy()
    df["dark"] = (df["skin"] > 0.5).astype(int)

    # Rates per game for descriptive summary.
    df["red_rate"] = df["redCards"] / df["games"]

    summary = df.groupby("dark").agg(
        n_dyads=("redCards", "size"),
        total_reds=("redCards", "sum"),
        total_games=("games", "sum"),
        mean_red_rate=("red_rate", "mean"),
    )
    summary["reds_per_game"] = summary["total_reds"] / summary["total_games"]

    # Poisson regression with exposure (games)
    X = sm.add_constant(df["dark"])
    y = df["redCards"]
    offset = np.log(df["games"])  # exposure

    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    res = model.fit()
    coef = res.params["dark"]
    se = res.bse["dark"]
    pval = res.pvalues["dark"]
    irr = np.exp(coef)

    print("Summary by skin tone (dark=1, light=0):")
    print(summary)
    print()
    print("Poisson GLM (offset log(games)):")
    print(f"coef_dark={coef:.4f}, SE={se:.4f}, IRR=exp(coef)={irr:.4f}, p-value={pval:.4g}")


if __name__ == "__main__":
    main()
