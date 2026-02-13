import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: ensure valid binomial counts
    df = df.copy()
    df = df[df["sockets"] > 0]

    # Indicator for modern humans (Homo sapiens) vs non-human primates
    df["is_human"] = df["genus"].str.contains("Homo", case=False, na=False).astype(int)

    # Proportion of teeth lost in this tooth class
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Drop any rows with missing covariates used in the model
    model_df = df[["prop_amtl", "is_human", "age", "prob_male", "tooth_class", "sockets"]].dropna()

    # Fit a binomial GLM with logit link:
    #   prop_amtl ~ is_human + age + prob_male (sex proxy) + tooth_class
    # using the number of observable sockets as binomial trial weights.
    glm_binom = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=model_df,
        family=sm.families.Binomial(),
        freq_weights=model_df["sockets"],
    )
    result = glm_binom.fit()

    # Extract effect of being human vs non-human primate
    coef_human = result.params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    p_human = result.pvalues.get("is_human", np.nan)

    # Compute an illustrative difference in predicted AMTL probability
    # for a "typical" specimen (mean age, mean prob_male, tooth_class=Anterior).
    mean_age = model_df["age"].mean()
    mean_prob_male = model_df["prob_male"].mean()

    # Build small design frame for prediction
    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": ["Anterior", "Anterior"],
        }
    )
    pred = result.get_prediction(pred_df)
    pred_means = np.asarray(pred.predicted_mean)
    prob_nonhuman, prob_human = float(pred_means[0]), float(pred_means[1])
    diff_prob = prob_human - prob_nonhuman

    # Map evidence strength to a 0–100 Likert-style response
    if np.isnan(coef_human) or np.isnan(p_human):
        response_scalar = 50
        interpretation = "Model could not estimate the human effect reliably; evidence is inconclusive."
    else:
        if coef_human > 0 and p_human < 0.001 and diff_prob > 0:
            response_scalar = 90
            interpretation = (
                "Strong evidence that modern humans have higher antemortem tooth loss frequencies "
                "than non-human primates after adjusting for age, sex, and tooth class."
            )
        elif coef_human > 0 and p_human < 0.05 and diff_prob > 0:
            response_scalar = 75
            interpretation = (
                "Moderate evidence that modern humans have higher antemortem tooth loss frequencies "
                "than non-human primates after adjusting for age, sex, and tooth class."
            )
        elif coef_human > 0 and p_human < 0.1 and diff_prob > 0:
            response_scalar = 65
            interpretation = (
                "Weak but suggestive evidence that modern humans have higher antemortem tooth loss frequencies "
                "than non-human primates after adjusting for age, sex, and tooth class."
            )
        elif coef_human <= 0 and p_human < 0.05 and diff_prob <= 0:
            response_scalar = 10
            interpretation = (
                "Evidence suggests modern humans do not have higher antemortem tooth loss frequencies "
                "than non-human primates once age, sex, and tooth class are controlled for."
            )
        else:
            response_scalar = 50
            interpretation = (
                "The estimated human effect on antemortem tooth loss is small or statistically uncertain; "
                "evidence for higher human AMTL frequencies is inconclusive after adjusting for covariates."
            )

    # Build explanation string with key numerical evidence
    explanation = (
        "I modeled the proportion of teeth lost (num_amtl/sockets) using a binomial regression with logit link, "
        "including an indicator for modern humans (Homo sapiens) versus non-human primates (Pan, Pongo, Papio), "
        "and adjusting for age, a probabilistic sex indicator (prob_male), and tooth class (anterior, posterior, premolar). "
        f"The estimated log-odds coefficient for humans was {coef_human:.3f} with standard error {se_human:.3f} "
        f"and p-value {p_human:.3g}. For a typical specimen (at the sample mean age and sex probability, anterior teeth), "
        f"the model predicted an antemortem tooth loss probability of {prob_nonhuman:.3f} for non-human primates and "
        f"{prob_human:.3f} for humans, a difference of {diff_prob:.3f}. "
        + interpretation
    )

    conclusion = {"response": int(response_scalar), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
