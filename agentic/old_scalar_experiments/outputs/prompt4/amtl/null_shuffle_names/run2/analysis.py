from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the AMTL dataset and construct semantically meaningful variables."""
    df = pd.read_csv(csv_path)

    # Map shuffled column names to their semantic roles based on info.json descriptions
    df = df.rename(
        columns={
            # Tooth region/category: Anterior / Posterior / Premolar
            "sockets": "tooth_region",
            # Numeric count of missing teeth of that class
            "genus": "num_missing",
            # Number of observable sockets that could be scored
            "age": "num_sockets",
            # Estimated age at death
            "pop": "age_years",
            # Probability specimen is male (originally `prob_male`)
            "stdev_age": "prob_male",
            # Specimen genus: Homo sapiens / Pan / Papio / Pongo
            "tooth_class": "genus_species",
        }
    )

    # Keep only rows with the genera of interest
    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus_species"].isin(target_genera)].copy()

    # Basic sanity filters
    df = df[
        (df["num_sockets"] > 0)
        & (df["num_missing"] >= 0)
        & (df["num_missing"] <= df["num_sockets"])
    ].copy()

    # Proportion of missing teeth (AMTL rate)
    df["amtl_prop"] = df["num_missing"] / df["num_sockets"]

    # Categorical encodings
    df["genus_species"] = df["genus_species"].astype("category")
    df["genus_species"] = df["genus_species"].cat.set_categories(
        ["Homo sapiens", "Pan", "Papio", "Pongo"], ordered=False
    )

    return df
 

def build_design_matrix(df: pd.DataFrame, design_columns=None) -> pd.DataFrame:
    """
    Construct the design matrix for the binomial model using one-hot encodings.

    Baseline categories (dropped):
      - genus_species: Homo sapiens
      - tooth_region: the first region alphabetically (e.g., Anterior)
    """
    base = df[["genus_species", "age_years", "prob_male", "tooth_region"]].copy()
    base["tooth_region"] = base["tooth_region"].astype("category")

    X = pd.get_dummies(base, drop_first=True)
    X = sm.add_constant(X, has_constant="add")

    if design_columns is not None:
        # Ensure new matrices have the same columns in the same order
        X = X.reindex(columns=design_columns, fill_value=0.0)

    return X


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial regression for AMTL proportion with weights = num_sockets."""
    X = build_design_matrix(df)
    y = df["amtl_prop"]

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result, X.columns


def compute_adjusted_rates(
    df: pd.DataFrame, result, design_columns
) -> dict:
    """
    Compute model-based, covariate-adjusted AMTL probabilities for each genus.

    For each genus, set genus_species to that value for all rows while holding
    age, sex (prob_male), and tooth_region at their observed values, then
    average the predicted probabilities (weighted by num_sockets).
    """
    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    adjusted = {}

    # Start from the original design matrix to preserve age, sex, and tooth_region effects
    X_base = build_design_matrix(df, design_columns)

    for g in genera:
        X_g = X_base.copy()

        # Encode genus via dummy variables with Homo sapiens as baseline (all zeros)
        for genus in ["Pan", "Papio", "Pongo"]:
            col = f"genus_species_{genus}"
            if col in X_g.columns:
                X_g[col] = 1.0 if genus == g else 0.0

        preds = result.predict(X_g)
        # Weight by number of sockets to approximate a socket-level average
        avg_prob = np.average(preds, weights=df["num_sockets"])
        adjusted[g] = float(avg_prob)

    return adjusted


def main():
    csv_path = Path("amtl.csv")
    if not csv_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory.")

    df = load_and_prepare_data(str(csv_path))

    # Descriptive AMTL proportions by genus (raw, unadjusted)
    raw_means = (
        df.groupby("genus_species")
        .apply(
            lambda g: (g["num_missing"].sum() / g["num_sockets"].sum())
        )
        .to_dict()
    )

    result, design_columns = fit_binomial_model(df)
    adjusted = compute_adjusted_rates(df, result, design_columns)

    # Print key outputs for inspection from the CLI
    print("Raw AMTL proportions by genus (missing teeth / observable sockets):")
    for g, p in raw_means.items():
        print(f"  {g:12s}: {p:.3f}")

    print("\nModel-based, covariate-adjusted AMTL probabilities by genus:")
    for g, p in adjusted.items():
        print(f"  {g:12s}: {p:.3f}")

    print("\nGenus coefficients (non-human genera relative to Homo sapiens):")
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"genus_species_{genus}"
        if term in result.params.index:
            coef = result.params[term]
            pval = result.pvalues[term]
            print(f"  {term}: coef = {coef:.3f}, p-value = {pval:.4g}")


if __name__ == "__main__":
    main()
