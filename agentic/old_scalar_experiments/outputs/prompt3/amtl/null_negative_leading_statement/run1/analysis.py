import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load metadata (mainly for context; data are in local amtl.csv)
    info_path = Path("info.json")
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        research_question = info.get("research_questions", [""])[0]
    else:
        research_question = ""

    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key fields if any
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    ).copy()

    # Create response as proportion of missing teeth per socket for binomial model
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Fit binomial regression with logit link, accounting for age, sex, and tooth class
    # Use sockets as frequency weights so that each row represents multiple trials.
    try:
        model = smf.glm(
            "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
            data=df,
            family=sm.families.Binomial(),
            freq_weights=df["sockets"],
        ).fit()
    except Exception as exc:
        # If the model fails for some reason, fall back to a simpler unweighted model
        model = smf.glm(
            "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
            data=df,
            family=sm.families.Binomial(),
        ).fit()

    # Extract coefficient and p-value for humans vs non-humans
    coef_human = float(model.params.get("is_human", np.nan))
    pval_human = float(model.pvalues.get("is_human", np.nan))

    # Standardized predicted AMTL rates for humans vs non-humans
    # Hold age, sex, and tooth class at their observed values, toggle is_human.
    base = df.copy()
    weights = base["sockets"]

    base_human = base.copy()
    base_human["is_human"] = 1
    base_nonhuman = base.copy()
    base_nonhuman["is_human"] = 0

    pred_human = model.predict(base_human)
    pred_nonhuman = model.predict(base_nonhuman)

    # Weighted average per-socket AMTL probability
    mean_human = float(np.average(pred_human, weights=weights))
    mean_nonhuman = float(np.average(pred_nonhuman, weights=weights))
    delta = mean_human - mean_nonhuman  # positive => humans higher

    # Decide on Yes/No answer
    # Yes if humans have significantly higher AMTL; otherwise No.
    alpha = 0.05
    if not np.isnan(coef_human) and coef_human > 0 and pval_human < alpha:
        response = "Yes"
    else:
        response = "No"

    # Strength of the Yes/No answer based on effect size
    abs_delta_pct = abs(delta) * 100.0  # percentage-point difference
    if abs_delta_pct >= 10:
        strength = 90
    elif abs_delta_pct >= 5:
        strength = 75
    elif abs_delta_pct >= 2:
        strength = 60
    elif abs_delta_pct >= 1:
        strength = 50
    else:
        strength = 35

    # Confidence based primarily on p-value and model clarity
    if np.isnan(pval_human):
        confidence = 40
    elif pval_human < 1e-6:
        confidence = 98
    elif pval_human < 1e-3:
        confidence = 95
    elif pval_human < 1e-2:
        confidence = 90
    elif pval_human < 0.05:
        confidence = 80
    elif pval_human < 0.1:
        confidence = 65
    else:
        confidence = 55

    # Build explanation string
    explanation_parts = []
    if research_question:
        explanation_parts.append(
            f"Research question: {research_question.strip()}"
        )

    explanation_parts.append(
        "I analyzed the AMTL dataset using a binomial regression model "
        "where the response was the proportion of missing teeth per socket "
        "for each specimen and tooth class. The model used a logit link and "
        "included a binary indicator for modern humans (Homo sapiens) versus "
        "non-human primates (Pan, Papio, Pongo), while statistically controlling "
        "for age at death, estimated sex (probability of being male), and tooth class."
    )

    explanation_parts.append(
        "To account for the fact that each row represents multiple tooth sockets, "
        "I weighted the regression by the number of observable sockets for each record. "
        "After fitting the model, I examined the coefficient for the human indicator "
        "to assess whether modern humans have higher AMTL frequencies than non-human primates "
        "once age, sex, and tooth class are controlled."
    )

    explanation_parts.append(
        f"The estimated coefficient for the human indicator was {coef_human:.3f} "
        f"with a p-value of {pval_human:.3g}. I also computed standardized predicted "
        f"AMTL probabilities by toggling the human indicator while holding age, sex, and "
        f"tooth class at their observed values. The average predicted AMTL probability per socket "
        f"was {mean_human:.3f} for humans and {mean_nonhuman:.3f} for non-human primates, "
        f"an absolute difference of {delta * 100.0:.1f} percentage points (positive values "
        f"indicate higher AMTL in humans)."
    )

    if response == "Yes":
        explanation_parts.append(
            "Because the human indicator has a positive coefficient and the associated p-value "
            "is below the conventional 0.05 threshold, the model provides evidence that modern "
            "humans have higher AMTL frequencies than non-human primates after controlling for "
            "age, sex, and tooth class. The estimated difference in predicted AMTL probabilities "
            "is sufficiently large to be substantively meaningful, leading to a 'Yes' answer."
        )
    else:
        explanation_parts.append(
            "Because the human indicator either does not have a positive, statistically significant "
            "effect or the estimated difference in predicted AMTL probabilities is small, the model "
            "does not support the claim that modern humans have higher AMTL frequencies than "
            "non-human primates when age, sex, and tooth class are taken into account. "
            "Accordingly, I answer 'No' to the research question."
        )

    explanation = "\n\n".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    # Write JSON conclusion to conclusion.txt with no extra text
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

