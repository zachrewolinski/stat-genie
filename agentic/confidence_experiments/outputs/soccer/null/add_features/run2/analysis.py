import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Skin tone: average of two raters when available
    df["skin_mean"] = df[["rater1", "rater2"]].mean(axis=1)

    # Keep rows with valid skin rating and games > 0
    df = df[df["skin_mean"].notna() & df["games"].notna() & (df["games"] > 0)].copy()

    # Define dark vs light: dark > 0.5, light < 0.5; drop neutral 0.5
    df = df[(df["skin_mean"] != 0.5)].copy()
    df["dark"] = (df["skin_mean"] > 0.5).astype(int)

    # Rate stats
    df["red_per_game"] = df["redCards"] / df["games"]

    summary = df.groupby("dark").agg(
        n=("redCards", "size"),
        total_red=("redCards", "sum"),
        total_games=("games", "sum"),
        mean_red_per_game=("red_per_game", "mean"),
        any_red_rate=("redCards", lambda x: (x > 0).mean()),
        skin_mean=("skin_mean", "mean"),
    )

    # Poisson regression with offset for games
    # Outcome: redCards, predictor: dark
    model_df = df[["redCards", "games", "dark"]].copy()
    model_df["intercept"] = 1.0
    poisson_model = sm.GLM(
        model_df["redCards"],
        model_df[["intercept", "dark"]],
        family=sm.families.Poisson(),
        offset=np.log(model_df["games"]),
    )
    poisson_res = poisson_model.fit(cov_type="HC0")

    # IRR for dark
    coef = poisson_res.params["dark"]
    se = poisson_res.bse["dark"]
    irr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
    pval = float(poisson_res.pvalues["dark"])

    # Linear rate difference (simple)
    # Average red per game by dark/light based on totals
    rate_dark = summary.loc[1, "total_red"] / summary.loc[1, "total_games"]
    rate_light = summary.loc[0, "total_red"] / summary.loc[0, "total_games"]
    rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan

    result = {
        "n_rows": int(len(df)),
        "summary": summary.reset_index().to_dict(orient="records"),
        "poisson": {
            "coef_dark": float(coef),
            "se_dark": float(se),
            "irr_dark": irr,
            "irr_ci_low": ci_low,
            "irr_ci_high": ci_high,
            "pvalue": pval,
        },
        "rate_dark": float(rate_dark),
        "rate_light": float(rate_light),
        "rate_ratio": float(rate_ratio),
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
