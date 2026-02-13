import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to match their semantic meaning from info.json
    df = df.rename(
        columns={
            "sockets": "tooth_type",  # Anterior / Posterior / Premolar
            "tooth_class": "genus_str",  # Homo sapiens / Pan / Papio / Pongo
            "age": "n_sockets",  # number of observable sockets
            "genus": "n_missing",  # number of missing teeth of given class
            "pop": "age_years",  # estimated age at death
            "stdev_age": "prob_male",  # proxy: probability specimen is male
            "prob_male": "specimen_id",  # unique specimen identifier
            "num_amtl": "age_uncertainty",
            "specimen": "region",
        }
    )

    # Basic cleaning: drop rows with non-positive sockets or missing key values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["n_sockets", "n_missing", "age_years", "prob_male", "tooth_type", "genus_str"]
    )
    df = df[df["n_sockets"] > 0]

    # Binary indicator for Homo sapiens vs. non-human primates.
    df["is_human"] = (df["genus_str"] == "Homo sapiens").astype(int)

    # Response as AMTL rate with binomial weights.
    df["amtl_rate"] = df["n_missing"] / df["n_sockets"]

    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression for AMTL rate with per-row socket counts as frequency weights.
    formula = "amtl_rate ~ is_human + age_years + prob_male + C(tooth_type)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )

    # Cluster-robust SEs by specimen to account for repeated measures.
    if "specimen_id" in df.columns:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen_id"]})
    else:
        result = model.fit()

    return result


def summarize_human_effect(df: pd.DataFrame, result) -> dict:
    coef = float(result.params["is_human"])
    conf_int = result.conf_int().loc["is_human"].tolist()
    p_value = float(result.pvalues["is_human"])

    # Average predicted AMTL probability per socket for humans vs non-humans,
    # using the same covariate distribution (standard marginal effect).
    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0

    preds_human = result.predict(df_human)
    preds_nonhuman = result.predict(df_nonhuman)

    avg_p_human = float(np.average(preds_human, weights=df["n_sockets"]))
    avg_p_nonhuman = float(np.average(preds_nonhuman, weights=df["n_sockets"]))

    return {
        "coef_is_human": coef,
        "conf_int_is_human": conf_int,
        "p_value_is_human": p_value,
        "avg_p_human": avg_p_human,
        "avg_p_nonhuman": avg_p_nonhuman,
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)
    result = fit_model(df)

    summary_stats = summarize_human_effect(df, result)

    # Print a compact summary for interactive inspection.
    print("is_human coefficient (log-odds):", summary_stats["coef_is_human"])
    print("95% CI for is_human:", summary_stats["conf_int_is_human"])
    print("p-value for is_human:", summary_stats["p_value_is_human"])
    print("Average predicted AMTL probability (human):", summary_stats["avg_p_human"])
    print("Average predicted AMTL probability (non-human):", summary_stats["avg_p_nonhuman"])


if __name__ == "__main__":
    main()

