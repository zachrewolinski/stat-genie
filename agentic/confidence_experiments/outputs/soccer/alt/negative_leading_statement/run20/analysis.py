import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.families.links import log as log_link


def main():
    df = pd.read_csv("soccer.csv")

    # Skin tone average (0=very light, 1=very dark)
    df["skin_avg"] = df[["rater1", "rater2"]].mean(axis=1)

    # Keep rows with needed data
    df = df.dropna(subset=["skin_avg", "redCards", "games"])

    # Binary light/dark split at midpoint (0.5 = neither light nor dark)
    df["dark"] = (df["skin_avg"] > 0.5).astype(int)

    # Summary rates per group
    group = df.groupby("dark").agg(
        n_rows=("dark", "size"),
        total_red=("redCards", "sum"),
        total_games=("games", "sum"),
        mean_red_per_dyad=("redCards", "mean"),
    )
    group["red_per_game"] = group["total_red"] / group["total_games"]

    # Poisson regression for redCards with offset log(games)
    # Predictor: dark (1=dark, 0=light)
    X = sm.add_constant(df["dark"])
    y = df["redCards"].astype(float)
    offset = np.log(df["games"].astype(float))

    model = sm.GLM(y, X, family=Poisson(link=log_link()), offset=offset)
    res = model.fit(cov_type="HC0")

    coef = res.params["dark"]
    se = res.bse["dark"]
    pval = res.pvalues["dark"]
    rr = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Additional: logistic regression on any red card (>=1)
    df["any_red"] = (df["redCards"] > 0).astype(int)
    X2 = sm.add_constant(df["dark"])
    y2 = df["any_red"].astype(float)
    logit_model = sm.Logit(y2, X2)
    logit_res = logit_model.fit(disp=False)
    logit_coef = logit_res.params["dark"]
    logit_se = logit_res.bse["dark"]
    logit_pval = logit_res.pvalues["dark"]
    logit_or = float(np.exp(logit_coef))
    logit_ci_low = float(np.exp(logit_coef - 1.96 * logit_se))
    logit_ci_high = float(np.exp(logit_coef + 1.96 * logit_se))

    print("GROUP_SUMMARY")
    print(group)
    print("\nPOISSON_RATE_RATIO")
    print({
        "rate_ratio": rr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(pval),
        "coef": float(coef),
        "se": float(se),
    })
    print("\nLOGIT_ANY_RED_ODDS_RATIO")
    print({
        "odds_ratio": logit_or,
        "ci_low": logit_ci_low,
        "ci_high": logit_ci_high,
        "p_value": float(logit_pval),
        "coef": float(logit_coef),
        "se": float(logit_se),
    })


if __name__ == "__main__":
    main()
