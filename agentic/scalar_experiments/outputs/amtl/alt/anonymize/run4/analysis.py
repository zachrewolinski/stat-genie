import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic sanity filters: keep rows with valid counts
    df = df.copy()
    df = df[(df["feature4"] > 0) & (df["feature3"] >= 0) & (df["feature3"] <= df["feature4"])]

    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = np.where(df["feature8"].str.contains("Homo", case=False), 1, 0)

    # Binomial response: proportion missing with trials as observable sockets
    df["missing_prop"] = df["feature3"] / df["feature4"]
    df["missing_prop"] = df["missing_prop"].clip(0.0, 1.0)

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Binomial regression:
    missing_prop ~ is_human + tooth class + age + sex
    with feature4 (observable sockets) as binomial trials (freq_weights).
    """
    formula = "missing_prop ~ is_human + C(feature1) + feature5 + feature7"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()
    return result


def compute_marginal_effects(result, df: pd.DataFrame):
    """
    Compute average predicted AMTL probability for humans vs non-humans,
    holding the empirical distribution of covariates fixed.
    """
    base = df.copy()

    nonhuman = base.copy()
    nonhuman["is_human"] = 0
    human = base.copy()
    human["is_human"] = 1

    pred_nonhuman = result.predict(nonhuman).mean()
    pred_human = result.predict(human).mean()
    diff = pred_human - pred_nonhuman

    return float(pred_human), float(pred_nonhuman), float(diff)


def main():
    df = load_data(Path("amtl.csv"))
    result = fit_binomial_model(df)

    coef_human = float(result.params.get("is_human", np.nan))
    se_human = float(result.bse.get("is_human", np.nan))
    pval_human = float(result.pvalues.get("is_human", np.nan))
    odds_ratio = float(np.exp(coef_human)) if np.isfinite(coef_human) else float("nan")

    pred_human, pred_nonhuman, diff = compute_marginal_effects(result, df)

    summary = {
        "coef_is_human": coef_human,
        "se_is_human": se_human,
        "pval_is_human": pval_human,
        "odds_ratio_is_human": odds_ratio,
        "pred_missing_prob_human": pred_human,
        "pred_missing_prob_nonhuman": pred_nonhuman,
        "pred_missing_prob_difference": diff,
        "n_rows": int(len(df)),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

