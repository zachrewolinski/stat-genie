import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    df = df.copy()
    # Drop rows with non-positive socket counts, though none are expected
    df = df[df["sockets"] > 0]
    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    # Overall AMTL rate by genus
    genus_group = df.groupby("genus").agg(
        total_missing=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    genus_group["amtl_rate"] = genus_group["total_missing"] / genus_group["total_sockets"]

    print("AMTL rate by genus (num_amtl / sockets):")
    print(genus_group.sort_values("amtl_rate", ascending=False))
    print()


def fit_binomial_glm(df: pd.DataFrame):
    # Use a binomial GLM with counts: successes = num_amtl, trials = sockets.
    # We model the log-odds of AMTL as a function of genus, age, sex (prob_male), and tooth class.
    # Set Homo sapiens as the reference genus to interpret coefficients for other genera.
    formula = "num_amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())
    print()

    return result


def predicted_probabilities_by_genus(result, df: pd.DataFrame) -> pd.DataFrame:
    # Compute adjusted predicted probabilities for each genus, holding covariates at their means
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    # Use the most common tooth_class as a representative category
    common_tooth_class = df["tooth_class"].mode().iloc[0]

    genera = sorted(df["genus"].unique())
    new_data = pd.DataFrame(
        {
            "genus": genera,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": common_tooth_class,
        }
    )

    preds = result.get_prediction(new_data)
    summary_frame = preds.summary_frame(alpha=0.05)

    out = new_data.copy()
    out["pred_prob"] = summary_frame["mean"]
    out["ci_lower"] = summary_frame["mean_ci_lower"]
    out["ci_upper"] = summary_frame["mean_ci_upper"]

    print("Adjusted predicted AMTL probabilities by genus")
    print(f"(age={mean_age:.2f}, prob_male={mean_prob_male:.2f}, tooth_class={common_tooth_class})")
    print(out.sort_values("pred_prob", ascending=False))
    print()

    return out


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "amtl.csv"

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows")
    print()

    descriptive_stats(df)
    glm_result = fit_binomial_glm(df)
    pred_df = predicted_probabilities_by_genus(glm_result, df)

    # Also print the genus-level ranking for quick manual inspection
    ranking = pred_df.sort_values("pred_prob", ascending=False)[
        ["genus", "pred_prob", "ci_lower", "ci_upper"]
    ]
    print("Genus ranking by adjusted AMTL probability (highest to lowest):")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()

