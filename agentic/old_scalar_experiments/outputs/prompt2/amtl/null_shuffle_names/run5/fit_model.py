import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load AMTL dataset and map columns to their semantic meanings."""
    df = pd.read_csv(csv_path)

    # Rename columns to their semantic roles based on info.json metadata.
    df = df.rename(
        columns={
            "sockets": "tooth_class",  # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",  # specimen identifier
            "genus": "num_missing",  # number of missing teeth in that class
            "age": "num_sockets",  # number of observable sockets
            "pop": "age_at_death",  # estimated age at death
            "num_amtl": "age_uncertainty",  # uncertainty in age estimate
            "stdev_age": "prob_male",  # probability specimen is male (0–1)
            "tooth_class": "genus",  # taxonomic genus: Homo sapiens, Pan, Papio, Pongo
            "specimen": "population",  # population / region label
        }
    )

    # Keep only variables needed for the regression.
    df = df[
        [
            "num_missing",
            "num_sockets",
            "age_at_death",
            "prob_male",
            "tooth_class",
            "genus",
        ]
    ].copy()

    # Ensure numeric columns are numeric.
    for col in ["num_missing", "num_sockets", "age_at_death", "prob_male"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing key covariates or invalid counts.
    df = df.dropna(subset=["num_missing", "num_sockets", "age_at_death", "prob_male"])
    df = df[df["num_sockets"] > 0]
    df = df[df["num_missing"] >= 0]
    df = df[df["num_missing"] <= df["num_sockets"]]

    # Indicator for modern humans vs non-human primates.
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # AMTL proportion response.
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit binomial regression of AMTL proportion on species and covariates."""
    formula = "prop_missing ~ human + age_at_death + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(result, df: pd.DataFrame) -> dict:
    """Summarize the human vs non-human effect on AMTL."""
    human_coef = float(result.params.get("human", np.nan))
    human_p = float(result.pvalues.get("human", np.nan))
    human_or = float(np.exp(human_coef)) if np.isfinite(human_coef) else np.nan

    # Predicted mean AMTL probability for humans vs non-humans,
    # averaging over the observed covariate distribution.
    base = df.copy()
    human_df = base.copy()
    human_df["human"] = 1
    nonhuman_df = base.copy()
    nonhuman_df["human"] = 0

    pred_human = float(result.predict(human_df).mean())
    pred_nonhuman = float(result.predict(nonhuman_df).mean())

    return {
        "human_coef": human_coef,
        "human_p": human_p,
        "human_or": human_or,
        "pred_human": pred_human,
        "pred_nonhuman": pred_nonhuman,
    }


def main():
    data_path = Path("amtl.csv")
    df = load_and_prepare_data(data_path)

    print("Prepared dataset shape:", df.shape)
    print(df[["genus", "tooth_class", "age_at_death", "prob_male"]].head())

    result = fit_binomial_model(df)
    print(result.summary())

    effect = summarize_effect(result, df)
    print("\n=== Human vs non-human AMTL effect summary ===")
    print(json.dumps(effect, indent=2))


if __name__ == "__main__":
    main()

