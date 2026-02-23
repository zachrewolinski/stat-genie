import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Ensure valid denominators
    df = df[df["sockets"] > 0].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial response: proportion missing with sockets as weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial regression controlling for age, sex (prob_male), and tooth class
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    # Cluster-robust SEs by specimen to account for repeated rows per specimen
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    # Extract key statistics for the human vs non-human contrast
    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    pval = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))
    ci_low, ci_high = np.exp(result.conf_int().loc["is_human"])

    # Marginal predicted probabilities averaged over the observed covariate distribution
    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0
    pred_human = float(result.predict(df_human).mean())
    pred_nonhuman = float(result.predict(df_nonhuman).mean())
    diff = pred_human - pred_nonhuman

    stats = {
        "coef_is_human": coef,
        "se_is_human": se,
        "pval_is_human": pval,
        "odds_ratio_is_human": odds_ratio,
        "odds_ratio_ci_low": float(ci_low),
        "odds_ratio_ci_high": float(ci_high),
        "predicted_mean_prop_human": pred_human,
        "predicted_mean_prop_nonhuman": pred_nonhuman,
        "predicted_mean_difference": diff,
        "n_rows": int(df.shape[0]),
        "n_specimens": int(df["specimen"].nunique()),
    }

    # Save numerical results for later interpretation
    with open("results.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()

