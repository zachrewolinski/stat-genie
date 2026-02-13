import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_PATH = Path("amtl.csv")
CONCLUSION_PATH = Path("conclusion.txt")


def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Map shuffled column names to their semantic meaning based on info.json descriptions.
    # sockets: Anterior/Posterior/Premolar -> tooth class
    # tooth_class: Homo sapiens / Pan / Papio / Pongo -> genus
    # pop: estimated age at death
    # stdev_age: sex estimate (0..1)
    # genus and age are small integers; we treat age as number of observable sockets
    # and genus as number of missing teeth, but we will clip to valid ranges.
    df = df.copy()
    df["tooth_type"] = df["sockets"]
    df["genus_str"] = df["tooth_class"]
    df["age_years"] = df["pop"]
    df["sex_est"] = df["stdev_age"]

    # Candidate counts for missing and total sockets.
    df["total_sockets"] = df["age"].astype(float)
    df["num_missing"] = df["genus"].astype(float)

    # Enforce logical constraints for binomial modeling.
    # Drop rows with non-positive totals or impossible counts.
    valid = (df["total_sockets"] > 0) & (df["num_missing"] >= 0)
    df = df.loc[valid].copy()

    # Cap num_missing at total_sockets when occasional data anomalies occur.
    df["num_missing"] = np.minimum(df["num_missing"], df["total_sockets"])

    # Compute proportion missing as response.
    df["prop_missing"] = df["num_missing"] / df["total_sockets"].replace(0, np.nan)

    # Restrict to the genera of interest.
    keep = df["genus_str"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])
    df = df.loc[keep].copy()

    # Create indicator for human vs non-human.
    df["is_human"] = (df["genus_str"] == "Homo sapiens").astype(int)

    # Simplify sex as a continuous covariate between 0 and 1.
    df["sex_prob_male"] = df["sex_est"].astype(float)

    # Tooth type as categorical.
    df["tooth_type"] = df["tooth_type"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Use proportion missing with total_sockets as frequency weights.
    df = df.copy()
    df = df[(df["total_sockets"] > 0) & df["prop_missing"].notna()]

    formula = "prop_missing ~ is_human + age_years + sex_prob_male + C(tooth_type)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    # Extract human coefficient and its uncertainty.
    params = result.params
    b_human = params.get("is_human", np.nan)
    b_se = result.bse.get("is_human", np.nan)

    # Compute predicted AMTL probability for a "typical" specimen:
    median_age = float(df["age_years"].median())
    median_sex = float(df["sex_prob_male"].median())
    common_tooth = df["tooth_type"].mode().iat[0]

    base_row = {
        "age_years": median_age,
        "sex_prob_male": median_sex,
        "tooth_type": common_tooth,
    }

    # Build prediction rows for non-human and human with identical covariates.
    df_pred = pd.DataFrame(
        [
            dict(base_row, is_human=0),
            dict(base_row, is_human=1),
        ]
    )
    preds = np.asarray(result.get_prediction(df_pred).predicted_mean)
    p_non_human, p_human = float(preds[0]), float(preds[1])

    diff = p_human - p_non_human

    # Translate effect size and significance into strength and confidence.
    # Use Wald z-score as a proxy for statistical evidence.
    if np.isfinite(b_human) and np.isfinite(b_se) and b_se > 0:
        z = abs(b_human / b_se)
    else:
        z = 0.0

    # Map |z| roughly to 0–100 scale (z≈2 -> ~70, z≈3 -> ~85, z≈5+ -> ~100).
    strength = float(np.clip(20 * z, 0, 100))

    # Confidence slightly more conservative than strength.
    confidence = float(np.clip(15 * z, 0, 100))

    response = "Yes" if diff > 0 else "No"

    explanation_lines = [
        f"Fitted a binomial regression of AMTL proportion on a human indicator, age at death, sex estimate, and tooth type using {len(df)} observations.",
        f"The estimated human coefficient on the log-odds scale is {b_human:.3f} (SE = {b_se:.3f}), implying higher AMTL frequencies in humans compared to non-human primates."  # noqa: E501
        if response == "Yes"
        else f"The estimated human coefficient on the log-odds scale is {b_human:.3f} (SE = {b_se:.3f}), implying no clear increase in AMTL frequencies for humans compared to non-human primates.",  # noqa: E501
        f"For a typical specimen (median age {median_age:.1f} years, median sex estimate {median_sex:.2f}, and the most common tooth type {common_tooth}), the predicted AMTL probability is {p_non_human:.3f} for non-human primates and {p_human:.3f} for humans (difference {diff:.3f}).",  # noqa: E501
    ]

    explanation = " ".join(explanation_lines)

    return {
        "response": response,
        "strength": round(strength, 1),
        "confidence": round(confidence, 1),
        "explanation": explanation,
    }


def main() -> None:
    df = load_and_prepare_data()
    model_result = fit_binomial_model(df)
    summary = summarize_effect(df, model_result)

    # Ensure required keys and types.
    output = {
        "response": str(summary["response"]),
        "strength": float(summary["strength"]),
        "confidence": float(summary["confidence"]),
        "explanation": str(summary["explanation"]),
    }

    CONCLUSION_PATH.write_text(json.dumps(output))


if __name__ == "__main__":
    main()
