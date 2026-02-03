import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("soccer.csv")

    # Skin tone as average of two raters
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)
    df = df.dropna(subset=["skin_mean", "redCards", "games"])
    df = df[df["games"] > 0]

    # Group summaries
    df["skin_group"] = pd.cut(
        df["skin_mean"],
        bins=[-0.01, 0.25, 0.75, 1.01],
        labels=["light", "medium", "dark"],
        include_lowest=True,
    )
    group_summary = (
        df.groupby("skin_group", observed=True)
        .agg(
            dyads=("skin_mean", "size"),
            total_games=("games", "sum"),
            total_reds=("redCards", "sum"),
        )
        .reset_index()
    )
    group_summary["reds_per_game"] = group_summary["total_reds"] / group_summary["total_games"]

    # Poisson regression with exposure offset
    X = sm.add_constant(df["skin_mean"])
    y = df["redCards"]
    offset = np.log(df["games"])
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    res = model.fit(cov_type="HC0")

    beta = res.params["skin_mean"]
    se = res.bse["skin_mean"]
    p = res.pvalues["skin_mean"]
    rate_ratio = float(np.exp(beta))

    # Predicted rates for light (0.25) and dark (0.75)
    exog = pd.DataFrame({"const": [1.0, 1.0], "skin_mean": [0.25, 0.75]})
    preds = res.predict(exog=exog, offset=np.array([0.0, 0.0]))
    pred_light = float(preds.iloc[0])
    pred_dark = float(preds.iloc[1])

    print("Group summary (dyads, total_games, total_reds, reds_per_game):")
    print(group_summary.to_string(index=False))
    print("\nPoisson regression: redCards ~ skin_mean with log(games) offset")
    print(res.summary())
    print(f"\nRate ratio for 1.0 increase in skin_mean: {rate_ratio:.3f}")
    print(f"Skin_mean coef: {beta:.4f}, SE: {se:.4f}, p-value: {p:.4g}")
    print(f"Predicted red-card rate per game at skin_mean=0.25: {pred_light:.5f}")
    print(f"Predicted red-card rate per game at skin_mean=0.75: {pred_dark:.5f}")


if __name__ == "__main__":
    main()
