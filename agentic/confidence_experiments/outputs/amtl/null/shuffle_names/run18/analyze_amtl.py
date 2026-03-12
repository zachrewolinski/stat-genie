import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Remap columns to their semantic meanings using info.json descriptions.
    # Original columns: sockets, prob_male, genus, age, pop, num_amtl, stdev_age, tooth_class, specimen
    # Semantics:
    # - Column "tooth_class" actually stores genus labels (Homo sapiens, Pan, Papio, Pongo)
    # - Column "sockets" stores tooth class (Anterior/Posterior/Premolar)
    # - Column "genus" is the count of missing teeth of that class
    # - Column "age" is the number of observable sockets
    # - Column "pop" is estimated age at death
    # - Column "stdev_age" is an estimate of sex (probability of male, 0–1)

    df["genus_label"] = df["tooth_class"]
    df["tooth_class"] = df["sockets"]
    df["num_missing"] = df["genus"].astype(int)
    df["num_sockets"] = df["age"].astype(int)
    df["age_years"] = df["pop"].astype(float)
    df["sex_prob_male"] = df["stdev_age"].astype(float)

    # Keep only rows with valid binomial counts: 0 <= missing <= sockets, sockets > 0
    df = df[df["num_sockets"] > 0].copy()
    df = df[(df["num_missing"] >= 0) & (df["num_missing"] <= df["num_sockets"])].copy()

    # Define human vs non-human primate indicator
    df["is_human"] = df["genus_label"].str.contains("Homo", case=False, na=False).astype(int)

    # Restrict to the four genera of interest
    valid_genera = {"Homo sapiens", "Homo", "Pan", "Papio", "Pongo"}
    df = df[df["genus_label"].isin(valid_genera)].copy()

    # Tooth class as categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Proportion of missing teeth per row
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression with logit link: AMTL proportion ~ human + age + sex + tooth class
    model = smf.glm(
        formula="prop_missing ~ is_human + age_years + sex_prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(result, df: pd.DataFrame) -> dict:
    coef_human = result.params.get("is_human", np.nan)
    p_human = result.pvalues.get("is_human", np.nan)
    odds_ratio_human = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    # Average predicted AMTL frequency for humans vs non-humans,
    # holding the distribution of other covariates fixed.
    df_human = df.copy()
    df_nonhuman = df.copy()
    df_human["is_human"] = 1
    df_nonhuman["is_human"] = 0
    pred_human = float(result.predict(df_human).mean())
    pred_nonhuman = float(result.predict(df_nonhuman).mean())

    return {
        "coef_human": float(coef_human),
        "p_human": float(p_human),
        "odds_ratio_human": odds_ratio_human,
        "pred_mean_human": pred_human,
        "pred_mean_nonhuman": pred_nonhuman,
        "pred_diff": pred_human - pred_nonhuman,
        "n_rows": int(df.shape[0]),
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_and_prepare_data(csv_path)

    print("Prepared data shape:", df.shape)
    print("Genus value counts:")
    print(df["genus_label"].value_counts())
    print("\nTooth class value counts:")
    print(df["tooth_class"].value_counts())

    result = fit_binomial_model(df)
    print("\nModel summary:")
    print(result.summary())

    summary = summarize_effect(result, df)
    print("\nHuman effect summary (controlling for age, sex, tooth class):")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

