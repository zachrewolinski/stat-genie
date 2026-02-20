import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to more meaningful semantic names based on metadata + inspection
    df = df.rename(
        columns={
            "tooth_class": "genus_label",  # actually holds Homo sapiens / Pan / Papio / Pongo
            "sockets": "tooth_class",  # Anterior / Posterior / Premolar
            "genus": "amtl_count",  # integer count of missing teeth in this class
            "age": "n_sockets_observed",  # integer count of observable sockets
            "pop": "age_at_death",  # estimated age at death
            "stdev_age": "sex_proxy",  # numeric proxy for sex (0–1 scale)
        }
    )

    # Drop any clearly inconsistent rows where missing teeth exceed observed sockets
    mask_valid = df["amtl_count"] <= df["n_sockets_observed"]
    df = df.loc[mask_valid].copy()

    # Ensure positive denominators
    df = df[df["n_sockets_observed"] > 0].copy()

    # Compute AMTL proportion for reference (model will use counts + weights)
    df["amtl_prop"] = df["amtl_count"] / df["n_sockets_observed"]

    # Center/scale continuous covariates for interpretation stability
    for col in ["age_at_death", "sex_proxy"]:
        mean_val = df[col].mean()
        df[f"{col}_c"] = df[col] - mean_val

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial GLM on proportions with the number of sockets as frequency weights
    formula = (
        "amtl_prop ~ C(genus_label) + age_at_death_c + sex_proxy_c + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets_observed"],
    )
    result = model.fit()
    return result


def compute_predicted_probs_by_genus(df: pd.DataFrame, result) -> pd.DataFrame:
    # Predict AMTL probability for each genus at mean covariate values and a reference tooth class
    mean_age = df["age_at_death_c"].mean()
    mean_sex = df["sex_proxy_c"].mean()

    # Use the most common tooth class as reference setting
    ref_tooth_class = df["tooth_class"].value_counts().idxmax()

    genera = sorted(df["genus_label"].unique())
    new_data = pd.DataFrame(
        {
            "genus_label": genera,
            "age_at_death_c": mean_age,
            "sex_proxy_c": mean_sex,
            "tooth_class": ref_tooth_class,
        }
    )
    preds = result.predict(new_data)
    new_data["pred_amtl_prob"] = preds
    return new_data


def main():
    df = load_data(Path("amtl.csv"))
    result = fit_binomial_model(df)

    print(result.summary())

    preds = compute_predicted_probs_by_genus(df, result)
    print("\nPredicted AMTL probability by genus (adjusted):")
    print(preds)

    # For downstream use in conclusion, save the key genus comparison to a JSON file
    # This is not required by the task but convenient for inspection if needed.
    output = {
        "predicted_probs": {
            row["genus_label"]: float(row["pred_amtl_prob"]) for _, row in preds.iterrows()
        }
    }
    Path("model_results.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

