import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats

DATA_PATH = "soccer.csv"


def cluster_stats(result, clusters):
    cov = cov_cluster(result, clusters)
    se = np.sqrt(np.diag(cov))
    params = result.params.values
    z = params / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_low = params + stats.norm.ppf(0.025) * se
    ci_high = params + stats.norm.ppf(0.975) * se
    return {
        "params": result.params.to_dict(),
        "se": dict(zip(result.params.index, se.tolist())),
        "pvalues": dict(zip(result.params.index, p.tolist())),
        "ci": {
            "0": dict(zip(result.params.index, ci_low.tolist())),
            "1": dict(zip(result.params.index, ci_high.tolist())),
        },
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # Skin tone as mean of two raters
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

    # Basic filtering
    df = df[(~df["skin_tone"].isna()) & (df["games"] > 0)].copy()

    # Define light/dark extremes based on 5-point scale mapped to [0,1]
    df["skin_group_extreme"] = pd.cut(
        df["skin_tone"],
        bins=[-np.inf, 0.25, 0.75, np.inf],
        labels=["light", "medium", "dark"],
    )

    # Define binary split at 0.5 as a robustness check
    df["skin_group_binary"] = np.where(df["skin_tone"] > 0.5, "dark", "light_or_mid")

    # Compute simple rates per game
    def rate_summary(group_col):
        grp = df.groupby(group_col).agg(
            red_cards=("redCards", "sum"),
            games=("games", "sum"),
            dyads=("redCards", "size"),
        )
        grp["red_per_game"] = grp["red_cards"] / grp["games"]
        return grp

    rate_extreme = rate_summary("skin_group_extreme")
    rate_binary = rate_summary("skin_group_binary")

    # Poisson regression with offset(log(games))
    # Model 1: continuous skin tone
    model_df = df[["redCards", "games", "skin_tone"]].copy()
    model_df["intercept"] = 1.0
    poisson_cont = sm.GLM(
        model_df["redCards"],
        model_df[["intercept", "skin_tone"]],
        family=sm.families.Poisson(),
        offset=np.log(model_df["games"]),
    ).fit()

    # Model 2: extreme dark vs light only
    extreme_df = df[df["skin_group_extreme"].isin(["light", "dark"])].copy()
    extreme_df["is_dark"] = (extreme_df["skin_group_extreme"] == "dark").astype(int)
    extreme_df["intercept"] = 1.0
    poisson_extreme = sm.GLM(
        extreme_df["redCards"],
        extreme_df[["intercept", "is_dark"]],
        family=sm.families.Poisson(),
        offset=np.log(extreme_df["games"]),
    ).fit()

    # Cluster-robust (by player and by referee) for extreme model
    cluster_player = cluster_stats(poisson_extreme, extreme_df["playerShort"])
    cluster_ref = cluster_stats(poisson_extreme, extreme_df["refNum"])

    # Output key stats
    out = {
        "n_total": int(len(df)),
        "n_extreme": int(len(extreme_df)),
        "rate_extreme": rate_extreme.reset_index().to_dict(orient="records"),
        "rate_binary": rate_binary.reset_index().to_dict(orient="records"),
        "poisson_cont_coef": poisson_cont.params.to_dict(),
        "poisson_cont_pvalues": poisson_cont.pvalues.to_dict(),
        "poisson_cont_ci": poisson_cont.conf_int().to_dict(),
        "poisson_extreme_coef": poisson_extreme.params.to_dict(),
        "poisson_extreme_pvalues": poisson_extreme.pvalues.to_dict(),
        "poisson_extreme_ci": poisson_extreme.conf_int().to_dict(),
        "poisson_extreme_cluster_player": cluster_player,
        "poisson_extreme_cluster_ref": cluster_ref,
    }

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
