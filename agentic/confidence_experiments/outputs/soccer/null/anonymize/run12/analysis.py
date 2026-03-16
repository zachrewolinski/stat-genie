import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Compute mean skin rating (0=very light, 1=very dark)
    df["skin_mean"] = df[["feature18", "feature19"]].mean(axis=1, skipna=True)

    # Keep rows with skin rating and games info
    df = df[df["skin_mean"].notna()].copy()
    df = df[df["feature9"].notna()].copy()

    # Outcome and exposure
    df["red_cards"] = df["feature16"].fillna(0)
    df["games"] = df["feature9"].astype(float)

    # Avoid zero games for offset
    df = df[df["games"] > 0].copy()

    # Unadjusted Poisson model with log(games) offset
    X = sm.add_constant(df["skin_mean"])
    model = sm.GLM(
        df["red_cards"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(df["games"]),
    )
    res = model.fit(cov_type="HC0")

    coef = res.params["skin_mean"]
    se = res.bse["skin_mean"]
    pval = res.pvalues["skin_mean"]
    rr = float(np.exp(coef))

    # Compare light vs dark groups (quartiles or thresholds)
    # Use quartiles to keep sample size reasonable
    q1 = df["skin_mean"].quantile(0.25)
    q3 = df["skin_mean"].quantile(0.75)
    light = df[df["skin_mean"] <= q1]
    dark = df[df["skin_mean"] >= q3]

    light_rate = light["red_cards"].sum() / light["games"].sum()
    dark_rate = dark["red_cards"].sum() / dark["games"].sum()
    rate_ratio = dark_rate / light_rate if light_rate > 0 else np.nan

    # Also compute predicted rates at skin_mean = 0 and 1
    intercept = res.params["const"]
    rate_light_pred = float(np.exp(intercept + coef * 0.0))
    rate_dark_pred = float(np.exp(intercept + coef * 1.0))
    pred_rate_ratio = rate_dark_pred / rate_light_pred if rate_light_pred > 0 else np.nan

    out = {
        "n_rows": int(len(df)),
        "skin_mean_min": float(df["skin_mean"].min()),
        "skin_mean_max": float(df["skin_mean"].max()),
        "coef_skin_mean": float(coef),
        "se_skin_mean": float(se),
        "pval_skin_mean": float(pval),
        "rate_ratio_poisson": rr,
        "light_rate": float(light_rate),
        "dark_rate": float(dark_rate),
        "light_vs_dark_rate_ratio": float(rate_ratio),
        "pred_rate_light": float(rate_light_pred),
        "pred_rate_dark": float(rate_dark_pred),
        "pred_rate_ratio": float(pred_rate_ratio),
        "q1_skin": float(q1),
        "q3_skin": float(q3),
        "light_n": int(len(light)),
        "dark_n": int(len(dark)),
    }

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
