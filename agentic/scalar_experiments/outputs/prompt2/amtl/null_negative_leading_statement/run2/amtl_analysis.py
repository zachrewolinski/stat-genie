import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: drop rows with missing key variables and nonpositive socket counts
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )
    df = df[df["sockets"] > 0]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial response as proportion with binomial weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Fit binomial (logistic) regression controlling for age, sex (prob_male), and tooth class
    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract human effect and summary statistics
    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    p_value = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Predicted AMTL proportions for humans vs non-humans
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    tooth_levels = sorted(df["tooth_class"].unique())

    pred_rows = [
        {
            "is_human": 1,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": tc,
        }
        for tc in tooth_levels
    ]
    pred_df_human = pd.DataFrame(pred_rows)
    pred_df_nonhuman = pred_df_human.copy()
    pred_df_nonhuman["is_human"] = 0

    prob_human = float(result.predict(pred_df_human).mean())
    prob_nonhuman = float(result.predict(pred_df_nonhuman).mean())
    diff_prob = prob_human - prob_nonhuman

    # Decide binary response and confidence
    if coef > 0 and p_value < 0.05:
        response = "Yes"
        if p_value < 0.001:
            confidence = 95
        elif p_value < 0.01:
            confidence = 90
        else:
            confidence = 80
    else:
        response = "No"
        if coef < 0 and p_value < 0.05:
            if p_value < 0.001:
                confidence = 95
            elif p_value < 0.01:
                confidence = 90
            else:
                confidence = 85
        elif p_value >= 0.05:
            if p_value < 0.1:
                confidence = 65
            else:
                confidence = 55
        else:
            confidence = 60

    confidence = int(max(0, min(100, round(confidence))))

    # Build explanation text
    explanation_parts = []
    explanation_parts.append(
        "We analyzed antemortem tooth loss (AMTL) in 1,450 specimen–tooth-class records "
        "from modern humans and three non-human primate genera (Pan, Pongo, Papio)."
    )
    explanation_parts.append(
        "For each record we modeled the number of missing teeth as a binomial outcome "
        "(num_amtl out of sockets) using logistic regression with predictors for a binary "
        "human-versus-non-human indicator, age at death, probability of being male, and tooth class."
    )
    explanation_parts.append(
        f"The estimated log-odds coefficient for humans was {coef:.3f} "
        f"(odds ratio {odds_ratio:.2f}, 95% CI {ci_low:.2f}–{ci_high:.2f}, p = {p_value:.3g})."
    )
    explanation_parts.append(
        "To interpret this effect on the probability scale, we used the fitted model to predict "
        "AMTL proportions at the mean age and sex across tooth classes."
    )
    explanation_parts.append(
        f"Under these conditions, the model-predicted AMTL proportion was {prob_human:.3f} for humans "
        f"and {prob_nonhuman:.3f} for non-human primates (difference {diff_prob:.3f})."
    )

    if response == "Yes":
        explanation_parts.append(
            "Because the human effect is positive and statistically significant, this model indicates "
            "that modern humans have higher AMTL frequencies than non-human primates after accounting "
            "for age, sex, and tooth class."
        )
    else:
        if coef < 0 and p_value < 0.05:
            explanation_parts.append(
                "Because the human effect is negative and statistically significant, this model indicates "
                "that modern humans actually have lower AMTL frequencies than non-human primates after "
                "accounting for age, sex, and tooth class."
            )
        elif p_value >= 0.05:
            explanation_parts.append(
                "Because the human effect is small relative to its uncertainty and not statistically "
                "distinguishable from zero, we do not find convincing evidence that modern humans have "
                "higher AMTL frequencies than non-human primates once age, sex, and tooth class are "
                "taken into account."
            )
        else:
            explanation_parts.append(
                "Overall, the data do not support the claim that modern humans have higher AMTL frequencies "
                "after adjusting for age, sex, and tooth class."
            )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

