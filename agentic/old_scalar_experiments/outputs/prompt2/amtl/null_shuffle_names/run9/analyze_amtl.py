import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata (for context / explanation if needed)
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    data_path = base_dir / "amtl.csv"
    df = pd.read_csv(data_path)

    # Rename columns to reflect their semantic meaning based on info.json
    df = df.rename(
        columns={
            "sockets": "tooth_region",      # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",     # Identifier
            "genus": "num_missing",         # Number of missing teeth of this class
            "age": "num_sockets",           # Number of observable sockets
            "pop": "age_at_death",          # Estimated age at death
            "num_amtl": "age_uncertainty",  # Uncertainty of age estimate
            "stdev_age": "sex_code",        # Encodes sex estimate
            "tooth_class": "genus",         # Taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
            "specimen": "region",           # Geographic region
        }
    )

    # Basic cleaning and sanity checks
    df = df.dropna(
        subset=["num_missing", "num_sockets", "age_at_death", "sex_code", "tooth_region", "genus"]
    )

    # Ensure counts are valid
    df = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0) & (df["num_missing"] <= df["num_sockets"])]

    # Compute response as proportion missing and set up weights (number of sockets)
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Fit binomial regression model for AMTL frequency
    # Model: AMTL proportion ~ human vs non-human + age + sex + tooth region
    formula = "prop_missing ~ is_human + age_at_death + sex_code + C(tooth_region)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()

    # Extract effect of being human
    coef_human = result.params.get("is_human", np.nan)
    pval_human = result.pvalues.get("is_human", np.nan)
    odds_ratio_human = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    # Predicted probabilities for human vs non-human at typical covariate values
    mean_age = float(df["age_at_death"].mean())
    mean_sex = float(df["sex_code"].mean())
    # Use the most common tooth region as a reference level
    mode_region = df["tooth_region"].mode().iloc[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_at_death": [mean_age, mean_age],
            "sex_code": [mean_sex, mean_sex],
            "tooth_region": [mode_region, mode_region],
        }
    )

    pred_res = result.get_prediction(pred_df).summary_frame()
    # The first row corresponds to non-human primates, the second to humans
    prob_nonhuman = float(pred_res["mean"].iloc[0])
    prob_human = float(pred_res["mean"].iloc[1])

    # Decide on Yes/No answer based on direction and significance of human effect
    if np.isnan(coef_human) or np.isnan(pval_human):
        response = "No"
        confidence = 40
        explanation = (
            "The regression model could not reliably estimate the effect of genus on AMTL frequency, "
            "so there is insufficient evidence to conclude that modern humans have higher AMTL rates than "
            "non-human primates after adjusting for age, sex, and tooth class."
        )
    else:
        # Positive coefficient means higher odds of AMTL for humans vs non-humans
        if coef_human > 0:
            response = "Yes"
        else:
            response = "No"

        # Map p-value to a heuristic confidence score
        if pval_human < 0.001:
            base_conf = 95
        elif pval_human < 0.01:
            base_conf = 90
        elif pval_human < 0.05:
            base_conf = 80
        elif pval_human < 0.1:
            base_conf = 65
        else:
            base_conf = 55

        confidence = base_conf

        explanation = (
            f"Research question: {research_question} "
            f"I fit a binomial regression model for the proportion of antemortem tooth loss per specimen "
            f"(number of missing teeth divided by observable sockets) using a logit link, with a binary predictor "
            f"indicating modern humans (Homo sapiens) versus non-human primates, and covariates for age at death, "
            f"sex estimate, and tooth class (anterior, posterior, premolar). "
            f"The coefficient for the human indicator was {coef_human:.3f}, corresponding to an odds ratio of "
            f"{odds_ratio_human:.2f} for AMTL in humans relative to non-human primates (p = {pval_human:.3g}). "
            f"At typical covariate values (mean age and sex, and the most common tooth region), the model predicts an "
            f"AMTL frequency of {prob_human:.3f} for humans and {prob_nonhuman:.3f} for non-human primates. "
            f"Based on the sign and statistical significance of the human effect after adjusting for age, sex, and "
            f"tooth class, the answer to the research question is '{response}'."
        )

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    # Write required JSON output to conclusion.txt with no extra text
    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

