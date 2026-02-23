import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Rename columns to meaningful names
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: ensure non-negative counts and sockets >= missing > =0
    df = df[(df["sockets"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets"])]

    # Create indicator for modern humans vs. non-human primates
    df["is_human"] = np.where(df["genus"].str.contains("Homo", case=False), 1, 0)

    # Proportion of missing teeth for binomial model
    df["prop_missing"] = df["missing"] / df["sockets"]

    # Drop rows with any missing values in key covariates
    df_model = df.dropna(
        subset=["prop_missing", "is_human", "age", "sex_estimate", "tooth_class"]
    ).copy()

    # Fit binomial regression: prop_missing ~ is_human + age + sex + tooth_class
    formula = "prop_missing ~ is_human + age + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Binomial(),
        freq_weights=df_model["sockets"],
    )
    result = model.fit()

    # Extract coefficient and p-value for is_human
    human_coef = result.params.get("is_human", np.nan)
    human_p = result.pvalues.get("is_human", np.nan)
    human_or = float(np.exp(human_coef)) if np.isfinite(human_coef) else np.nan

    # Compute a simple effect size in probability space at median covariates
    med_age = df_model["age"].median()
    med_sex = df_model["sex_estimate"].median()
    # Use the most common tooth class as reference
    ref_tooth = df_model["tooth_class"].mode().iat[0]

    def predict_prob(is_human_flag: int) -> float:
        # Build a small DataFrame for prediction
        pred_df = pd.DataFrame(
            {
                "is_human": [is_human_flag],
                "age": [med_age],
                "sex_estimate": [med_sex],
                "tooth_class": [ref_tooth],
            }
        )
        pred = result.predict(pred_df)
        return float(pred.iloc[0])

    prob_nonhuman = predict_prob(0)
    prob_human = predict_prob(1)
    prob_diff = prob_human - prob_nonhuman

    # Print key results for inspection in the CLI
    print("Binomial regression results (AMTL proportion):")
    print(result.summary())
    print("\nKey human effect metrics:")
    print(f"is_human coefficient (log-odds): {human_coef:.4f}")
    print(f"is_human odds ratio: {human_or:.3f}")
    print(f"is_human p-value: {human_p:.3e}")
    print(f"Predicted AMTL probability (non-human): {prob_nonhuman:.4f}")
    print(f"Predicted AMTL probability (human): {prob_human:.4f}")
    print(f"Difference in probability (human - non-human): {prob_diff:.4f}")

    # Save numeric summaries so they can be reused for the final explanation
    summary = {
        "human_coef": human_coef,
        "human_or": human_or,
        "human_p": human_p,
        "prob_nonhuman": prob_nonhuman,
        "prob_human": prob_human,
        "prob_diff": prob_diff,
    }
    with open("analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

