import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    # Remap scrambled column names to their semantic meaning based on info.json.
    df = df_raw.copy()
    df["tooth_position"] = df_raw["sockets"]  # Anterior / Posterior / Premolar
    df["specimen_id"] = df_raw["prob_male"]
    df["num_missing"] = df_raw["genus"]  # number of missing teeth of given class
    df["num_sockets"] = df_raw["age"]  # observable sockets that could be scored
    df["age_at_death"] = df_raw["pop"]
    df["age_uncertainty"] = df_raw["num_amtl"]
    df["prob_male"] = df_raw["stdev_age"]  # 0=female, 1=male, 0.5=unknown
    df["genus"] = df_raw["tooth_class"]  # Homo sapiens, Pan, Papio, Pongo
    df["population"] = df_raw["specimen"]

    # Keep only the genera relevant to the question.
    valid_genera = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus"].isin(valid_genera)].copy()

    # Ensure integer counts and valid ranges.
    df["num_missing"] = df["num_missing"].astype(int)
    df["num_sockets"] = df["num_sockets"].astype(int)
    df = df[df["num_sockets"] > 0].copy()
    df["num_missing"] = df[["num_missing", "num_sockets"]].min(axis=1)

    return df


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    # Create one row per observable socket, with a binary AMTL outcome.
    records = []
    for _, row in df.iterrows():
        n_total = int(row["num_sockets"])
        n_missing = int(row["num_missing"])
        n_missing = max(0, min(n_missing, n_total))

        amtl_values = np.array([1] * n_missing + [0] * (n_total - n_missing), dtype=int)

        records.append(
            pd.DataFrame(
                {
                    "amtl": amtl_values,
                    "genus": row["genus"],
                    "age_at_death": row["age_at_death"],
                    "prob_male": row["prob_male"],
                    "tooth_position": row["tooth_position"],
                }
            )
        )

    df_long = pd.concat(records, ignore_index=True)
    df_long["is_human"] = (df_long["genus"] == "Homo sapiens").astype(int)
    return df_long


def fit_model(df_long: pd.DataFrame):
    # Logistic regression: AMTL ~ human vs non-human + age + sex + tooth position
    model = smf.logit(
        "amtl ~ is_human + age_at_death + prob_male + C(tooth_position)",
        data=df_long,
    ).fit(disp=False)
    return model


def summarize_effect(model):
    coef = model.params["is_human"]
    pval = model.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))
    return float(coef), float(pval), odds_ratio


def derive_conclusion(coef: float, pval: float, odds_ratio: float):
    # Positive coefficient => humans have higher odds of AMTL.
    # Use p-value and effect size to set a heuristic confidence.
    if coef > 0:
        response = "Yes"
    else:
        response = "No"

    # Confidence heuristic.
    if pval < 1e-6 and (odds_ratio > 1.3 or odds_ratio < 1 / 1.3):
        base_conf = 95
    elif pval < 1e-3:
        base_conf = 85
    elif pval < 0.01:
        base_conf = 75
    elif pval < 0.05:
        base_conf = 65
    else:
        base_conf = 55

    # Slightly boost confidence for larger absolute effects.
    abs_coef = abs(coef)
    if abs_coef > 0.75:
        base_conf += 5
    if abs_coef > 1.0:
        base_conf += 5

    confidence = max(0, min(100, base_conf))

    return response, confidence


def build_explanation(coef: float, pval: float, odds_ratio: float) -> str:
    direction = "higher" if coef > 0 else "lower"
    explanation = (
        "I fitted a logistic regression model at the tooth level "
        "where each observable socket was coded as either present or missing (AMTL=1). "
        "The outcome (AMTL) was modeled as a function of a binary indicator for Homo sapiens "
        "versus non-human primates (Pan, Pongo, Papio), while adjusting for estimated age at death, "
        "probability of being male, and tooth class (anterior, premolar, posterior). "
        f"In this model the coefficient for the human indicator was {coef:.3f}, which corresponds to an odds ratio of {odds_ratio:.2f} "
        f"and a p-value of {pval:.2e}. "
        f"This indicates that, after accounting for age, sex, and tooth class, modern humans have {direction} odds of antemortem tooth loss "
        "compared to the pooled non-human primates in the dataset."
    )
    return explanation


def main():
    df = load_and_prepare_data("amtl.csv")
    df_long = expand_to_tooth_level(df)
    model = fit_model(df_long)
    coef, pval, odds_ratio = summarize_effect(model)
    response, confidence = derive_conclusion(coef, pval, odds_ratio)
    explanation = build_explanation(coef, pval, odds_ratio)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()
