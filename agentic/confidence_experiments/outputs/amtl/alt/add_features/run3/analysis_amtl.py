import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_csv(data_path)

    # Keep only the genera relevant to the research question.
    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Basic cleaning: drop rows with missing key fields and require positive socket counts.
    key_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    df = df.dropna(subset=key_cols)
    df = df[df["sockets"] > 0].copy()

    # Construct binomial response: proportion of missing teeth with socket count as weight.
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Binary indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial regression with logit link, weighting by number of sockets.
    # We model AMTL rate as a function of human vs non-human, age, sex proxy, and tooth class.
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract the human vs non-human effect.
    if "is_human" not in result.params:
        raise RuntimeError("Model did not include is_human term as expected.")

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Compute representative predicted AMTL probabilities for human vs non-human
    # at typical covariate values (mean age, mean prob_male, posterior teeth).
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    # Use the most common tooth class as representative.
    tooth_class_mode = df["tooth_class"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [tooth_class_mode, tooth_class_mode],
        }
    )
    predicted_rates = result.predict(pred_df)
    nonhuman_rate, human_rate = map(float, predicted_rates)

    # Save a compact JSON summary of the core statistical evidence to inspect manually.
    summary = {
        "n_rows_used": int(df.shape[0]),
        "genera_counts": df["genus"].value_counts().to_dict(),
        "tooth_class_counts": df["tooth_class"].value_counts().to_dict(),
        "coef_is_human": coef,
        "pval_is_human": pval,
        "odds_ratio_is_human": odds_ratio,
        "representative_tooth_class": tooth_class_mode,
        "predicted_amtl_rate_nonhuman": nonhuman_rate,
        "predicted_amtl_rate_human": human_rate,
    }

    with open("analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also print a brief human-readable summary to stdout for interactive inspection.
    print("Binomial regression (AMTL rate) summary:")
    print(f"Rows used: {summary['n_rows_used']}")
    print("Genus counts:", summary["genera_counts"])
    print("Tooth class counts:", summary["tooth_class_counts"])
    print(f"is_human coefficient: {coef:.4f}")
    print(f"is_human odds ratio: {odds_ratio:.3f}")
    print(f"is_human p-value: {pval:.4g}")
    print(
        "Representative predicted AMTL rate at mean age/sex, "
        f"{tooth_class_mode} teeth:"
    )
    print(f"  Non-human primates: {nonhuman_rate:.4f}")
    print(f"  Modern humans:      {human_rate:.4f}")


if __name__ == "__main__":
    main()

