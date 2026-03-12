import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"


def poisson_glm_cluster(df, y_col, x_cols, offset_col=None, cluster_col=None):
    X = df[x_cols]
    X = sm.add_constant(X, has_constant="add")
    y = df[y_col]
    offset = None
    if offset_col is not None:
        offset = df[offset_col]
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    if cluster_col is not None:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    else:
        res = model.fit()
    return res


def binomial_glm_cluster(df, y_col, x_cols, cluster_col=None):
    X = df[x_cols]
    X = sm.add_constant(X, has_constant="add")
    y = df[y_col]
    model = sm.GLM(y, X, family=sm.families.Binomial())
    if cluster_col is not None:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    else:
        res = model.fit()
    return res


def main():
    df = pd.read_csv(DATA_PATH)

    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
    df["log_games"] = np.log(df["games"].astype(float))
    df["any_red"] = (df["redCards"] > 0).astype(int)

    df_skin = df.dropna(subset=["skin_tone"]).copy()

    # Define light/dark groups based on 5-point scale (0, 0.25, 0.5, 0.75, 1.0)
    df_skin["dark"] = (df_skin["skin_tone"] >= 0.75).astype(int)
    df_skin["light"] = (df_skin["skin_tone"] <= 0.25).astype(int)
    df_dark_light = df_skin[(df_skin["dark"] == 1) | (df_skin["light"] == 1)].copy()
    df_dark_light["dark"] = (df_dark_light["dark"] == 1).astype(int)

    summary = {}
    summary["n_total"] = int(len(df))
    summary["n_with_skin"] = int(len(df_skin))
    summary["n_dark_light"] = int(len(df_dark_light))

    def group_stats(sub):
        return {
            "n_dyads": int(len(sub)),
            "total_games": float(sub["games"].sum()),
            "total_red_cards": float(sub["redCards"].sum()),
            "red_cards_per_game": float(sub["redCards"].sum() / sub["games"].sum()),
            "any_red_rate": float(sub["any_red"].mean()),
        }

    summary["light"] = group_stats(df_dark_light[df_dark_light["dark"] == 0])
    summary["dark"] = group_stats(df_dark_light[df_dark_light["dark"] == 1])

    # Poisson model: redCards ~ dark + offset(log_games)
    if len(df_dark_light) > 0:
        res_pois = poisson_glm_cluster(
            df_dark_light,
            y_col="redCards",
            x_cols=["dark"],
            offset_col="log_games",
            cluster_col="playerShort",
        )
        coef = res_pois.params["dark"]
        se = res_pois.bse["dark"]
        pval = res_pois.pvalues["dark"]
        irr = float(np.exp(coef))
        summary["poisson_dark_vs_light"] = {
            "coef": float(coef),
            "se": float(se),
            "pval": float(pval),
            "irr": irr,
        }

        # Binomial model for any red card (with log_games as covariate)
        res_bin = binomial_glm_cluster(
            df_dark_light,
            y_col="any_red",
            x_cols=["dark", "log_games"],
            cluster_col="playerShort",
        )
        coef_b = res_bin.params["dark"]
        se_b = res_bin.bse["dark"]
        pval_b = res_bin.pvalues["dark"]
        or_b = float(np.exp(coef_b))
        summary["logit_any_red_dark_vs_light"] = {
            "coef": float(coef_b),
            "se": float(se_b),
            "pval": float(pval_b),
            "or": or_b,
        }

    # Continuous skin tone model using all rows with skin tone
    if len(df_skin) > 0:
        res_pois_cont = poisson_glm_cluster(
            df_skin,
            y_col="redCards",
            x_cols=["skin_tone"],
            offset_col="log_games",
            cluster_col="playerShort",
        )
        coef_c = res_pois_cont.params["skin_tone"]
        se_c = res_pois_cont.bse["skin_tone"]
        pval_c = res_pois_cont.pvalues["skin_tone"]
        irr_c = float(np.exp(coef_c))
        summary["poisson_continuous_skin_tone"] = {
            "coef": float(coef_c),
            "se": float(se_c),
            "pval": float(pval_c),
            "irr_per_unit": irr_c,
        }

        res_bin_cont = binomial_glm_cluster(
            df_skin,
            y_col="any_red",
            x_cols=["skin_tone", "log_games"],
            cluster_col="playerShort",
        )
        coef_bc = res_bin_cont.params["skin_tone"]
        se_bc = res_bin_cont.bse["skin_tone"]
        pval_bc = res_bin_cont.pvalues["skin_tone"]
        or_bc = float(np.exp(coef_bc))
        summary["logit_any_red_continuous_skin_tone"] = {
            "coef": float(coef_bc),
            "se": float(se_bc),
            "pval": float(pval_bc),
            "or_per_unit": or_bc,
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
