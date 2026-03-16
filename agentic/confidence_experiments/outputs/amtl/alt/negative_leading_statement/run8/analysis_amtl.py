import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Create outcome as proportion of missing teeth with binomial denominator.
    df = df.copy()
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans versus non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Center age for stability.
    df["age_c"] = df["age"] - df["age"].mean()

    # Use prob_male as a continuous proxy for sex.
    # Tooth class as categorical.
    formula = "prop_missing ~ is_human + age_c + prob_male + C(tooth_class)"

    # Because the response is a proportion with known denominators, fit a binomial GLM with weights.
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract key statistics for the human vs non-human contrast.
    human_coef = result.params["is_human"]
    human_se = result.bse["is_human"]
    human_p = result.pvalues["is_human"]

    # Convert coefficient to odds ratio.
    human_or = float(np.exp(human_coef))

    # Map statistical evidence to Likert response.
    # Strong evidence for higher human AMTL if OR>1 and p<0.001.
    # Moderate if p<0.01, weak if p<0.05, otherwise essentially no evidence.
    if human_p < 0.001 and human_or > 1:
        response = 90
        qualitative = "strong"
        direction = "higher"
    elif human_p < 0.01 and human_or > 1:
        response = 75
        qualitative = "moderate"
        direction = "higher"
    elif human_p < 0.05 and human_or > 1:
        response = 60
        qualitative = "weak"
        direction = "higher"
    elif human_p < 0.05 and human_or < 1:
        # Statistically significant but in the opposite direction.
        response = 20
        qualitative = "weak"
        direction = "lower"
    elif human_p < 0.01 and human_or < 1:
        response = 10
        qualitative = "moderate"
        direction = "lower"
    elif human_p < 0.001 and human_or < 1:
        response = 5
        qualitative = "strong"
        direction = "lower"
    else:
        # No clear evidence for a difference.
        response = 40
        qualitative = "little to no"
        direction = "different"

    # Construct explanation.
    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) after "
        "adjusting for age, sex, and tooth class?\n\n"
        "Method: I modeled the proportion of missing teeth (num_amtl / sockets) using a "
        "binomial generalized linear model with a logit link. The model included a binary "
        "indicator for modern humans vs. non-human primates, centered age, the estimated "
        "probability of being male, and tooth class (Anterior, Posterior, Premolar) as a "
        "categorical predictor. Each row was weighted by the number of observable sockets "
        "to account for varying denominators.\n\n"
        f"Key result: The coefficient for the human indicator corresponds to an odds ratio "
        f"of approximately {human_or:.2f} for AMTL when comparing modern humans to "
        "non-human primates, holding age, sex, and tooth class constant. The associated "
        f"p-value is {human_p:.3g}, indicating {qualitative} statistical evidence that "
        f"modern humans have {direction} AMTL frequencies than non-human primates.\n\n"
        "Interpretation: Based on this model, the evidence "
    )

    if direction == "higher":
        explanation += (
            "supports the conclusion that modern humans exhibit higher AMTL frequencies "
            "than non-human primates after accounting for age, sex, and tooth class."
        )
    elif direction == "lower":
        explanation += (
            "suggests that modern humans actually have lower AMTL frequencies than "
            "non-human primates once age, sex, and tooth class are controlled for, contrary "
            "to the initial 'No' expectation."
        )
    else:
        explanation += (
            "does not provide clear evidence for a difference in AMTL frequencies between "
            "modern humans and non-human primates after adjusting for age, sex, and tooth "
            "class; estimates are compatible with similar AMTL rates across genera."
        )

    explanation += (
        f" I mapped this evidence to a 0–100 Likert scale, where 0 is a strong 'No' and "
        f"100 is a strong 'Yes', yielding a response value of {response}."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required conclusion file.
    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

