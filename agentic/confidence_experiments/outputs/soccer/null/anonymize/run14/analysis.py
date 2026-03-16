import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("soccer.csv")
    # Compute mean skin tone across raters
    df["skin_mean"] = df[["feature18", "feature19"]].mean(axis=1)
    # Keep rows with skin ratings and games > 0
    df = df[df["skin_mean"].notna() & (df["feature9"] > 0)].copy()
    df["games"] = df["feature9"].astype(float)
    df["red_cards"] = df["feature16"].astype(float)

    # Poisson GLM for red cards with exposure offset (games)
    X = sm.add_constant(df["skin_mean"])
    model = sm.GLM(
        df["red_cards"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(df["games"]),
    )
    res = model.fit(cov_type="HC0")
    coef = res.params["skin_mean"]
    pval = res.pvalues["skin_mean"]
    irr = float(np.exp(coef))

    # Predicted rate per game at light vs dark skin values
    rate_light = float(np.exp(res.params["const"] + coef * 0.25))
    rate_dark = float(np.exp(res.params["const"] + coef * 0.75))

    # Discrete comparison for very light vs very dark
    sub = df[(df["skin_mean"] <= 0.25) | (df["skin_mean"] >= 0.75)].copy()
    sub["dark"] = (sub["skin_mean"] >= 0.75).astype(int)
    X2 = sm.add_constant(sub["dark"])
    model2 = sm.GLM(
        sub["red_cards"],
        X2,
        family=sm.families.Poisson(),
        offset=np.log(sub["games"]),
    )
    res2 = model2.fit(cov_type="HC0")
    irr_dark = float(np.exp(res2.params["dark"]))
    pval_dark = res2.pvalues["dark"]

    # Simple aggregated rates
    light = df[df["skin_mean"] <= 0.25]
    dark = df[df["skin_mean"] >= 0.75]
    rate_light_agg = float(light["red_cards"].sum() / light["games"].sum())
    rate_dark_agg = float(dark["red_cards"].sum() / dark["games"].sum())
    rate_ratio_agg = float(rate_dark_agg / rate_light_agg) if rate_light_agg > 0 else float("nan")

    out = {
        "n_rows": int(df.shape[0]),
        "n_light": int(light.shape[0]),
        "n_dark": int(dark.shape[0]),
        "poisson_coef": float(coef),
        "poisson_irr": irr,
        "poisson_p": float(pval),
        "rate_light_pred": rate_light,
        "rate_dark_pred": rate_dark,
        "dark_vs_light_irr": irr_dark,
        "dark_vs_light_p": float(pval_dark),
        "rate_light_agg": rate_light_agg,
        "rate_dark_agg": rate_dark_agg,
        "rate_ratio_agg": rate_ratio_agg,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
