import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Restrict to relevant genera and drop rows with missing key fields
    relevant_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(relevant_genera)].copy()
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Basic sanity checks
    df = df[df["sockets"] > 0].copy()
    df = df[df["num_amtl"] >= 0].copy()
    df = df[df["num_amtl"] <= df["sockets"]].copy()

    # Indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of sockets that are missing for binomial modeling
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    print("=== Dataset overview ===")
    print("Rows:", len(df))
    print("Genus counts:")
    print(df["genus"].value_counts())
    print("\nMean AMTL proportion by genus (simple averages across specimens):")
    print(df.groupby("genus")["prop_amtl"].mean())

    # Binomial regression: AMTL proportion as a function of human vs non-human,
    # age, sex (prob_male), and tooth class, with sockets as binomial weights.
    print("\n=== Binomial regression (GLM, logit link) ===")
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    print(result.summary())

    # Compute predicted AMTL probabilities for humans vs non-humans at typical values.
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    # Use the most common tooth class as a representative category.
    common_tooth_class = df["tooth_class"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [common_tooth_class, common_tooth_class],
        }
    )

    preds = result.get_prediction(pred_df).summary_frame()
    pred_nonhuman = preds["mean"].iloc[0]
    pred_human = preds["mean"].iloc[1]

    print("\n=== Predicted AMTL probability at representative values ===")
    print(f"Representative tooth class: {common_tooth_class}")
    print(f"Mean age: {mean_age:.2f}, mean prob_male: {mean_prob_male:.2f}")
    print(f"Non-human primates (Pan/Pongo/Papio): {pred_nonhuman:.3f}")
    print(f"Modern humans (Homo sapiens):       {pred_human:.3f}")

    # Save a small JSON summary of key numerical results for inspection if needed.
    summary = {
        "mean_age": float(mean_age),
        "mean_prob_male": float(mean_prob_male),
        "common_tooth_class": common_tooth_class,
        "pred_nonhuman": float(pred_nonhuman),
        "pred_human": float(pred_human),
        "coef_is_human": float(result.params.get("is_human", np.nan)),
        "pvalue_is_human": float(result.pvalues.get("is_human", np.nan)),
    }
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved numerical summary to analysis_summary.json")


if __name__ == "__main__":
    main()

