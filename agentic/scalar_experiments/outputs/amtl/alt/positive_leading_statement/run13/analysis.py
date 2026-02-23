import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic sanity: drop any rows with missing key fields (should be rare/non-existent)
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Create indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth for descriptives
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Descriptive: overall AMTL rates by genus
    genus_summary = (
        df.groupby("genus")
        .apply(lambda g: pd.Series({
            "total_missing": g["num_amtl"].sum(),
            "total_sockets": g["sockets"].sum(),
        }))
        .assign(rate=lambda x: x["total_missing"] / x["total_sockets"])
    )

    # Binomial regression: AMTL rate ~ human vs non-human + age + sex + tooth class
    # Use grouped-binomial form with sockets as frequency weights
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract key statistics for the human effect
    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    p_human = float(result.pvalues["is_human"])
    or_human = float(np.exp(coef_human))

    # Compute predicted AMTL probabilities for a "typical" case
    # (median age, prob_male=0.5, tooth_class='Anterior'), for human vs non-human
    typical = {
        "age": df["age"].median(),
        "prob_male": 0.5,
        "tooth_class": "Anterior",
    }
    pred_df = pd.DataFrame([
        dict(typical, is_human=0),
        dict(typical, is_human=1),
    ])
    pred = result.get_prediction(pred_df).summary_frame()
    p_nonhuman = float(pred.loc[0, "mean"])
    p_human_pred = float(pred.loc[1, "mean"])

    # Map evidence strength to Likert 0-100 scale.
    # We consider both statistical significance and effect size (odds ratio and absolute difference).
    if p_human < 0.001 and or_human > 1.2 and (p_human_pred - p_nonhuman) > 0.03:
        likert = 95
    elif p_human < 0.01 and or_human > 1.1 and (p_human_pred - p_nonhuman) > 0.02:
        likert = 85
    elif p_human < 0.05 and or_human > 1.05 and (p_human_pred - p_nonhuman) > 0.01:
        likert = 70
    elif p_human < 0.05 and or_human > 1.0:
        likert = 60
    elif p_human < 0.1 and or_human > 1.0:
        likert = 55
    else:
        # No convincing evidence that humans have higher AMTL
        if p_human >= 0.1:
            likert = 20
        else:
            likert = 40

    # Build explanation string summarizing analysis and results
    explanation_parts = []

    # Overall descriptive rates
    desc_lines = []
    for genus, row in genus_summary.iterrows():
        desc_lines.append(
            f"{genus}: {row['total_missing']:.0f} missing of {row['total_sockets']:.0f} sockets "
            f"({row['rate'] * 100:.1f}% missing)."
        )

    explanation_parts.append(
        "I analyzed the antemortem tooth loss (AMTL) dataset using a binomial regression model "
        "with the proportion of missing teeth (num_amtl / sockets) as the response and predictors "
        "for whether a specimen was a modern human (Homo sapiens) versus a non-human primate, "
        "along with age at death, estimated sex (prob_male), and tooth class (anterior, posterior, premolar)."
    )
    explanation_parts.append(
        "Descriptively, the overall AMTL rates by genus (missing teeth divided by total observable "
        "sockets) were: " + " ".join(desc_lines)
    )
    explanation_parts.append(
        f"In the regression model, the coefficient for the human indicator (Homo sapiens vs. non-human primates) "
        f"was {coef_human:.3f} (SE {se_human:.3f}), corresponding to an odds ratio of {or_human:.2f} for AMTL "
        f"in humans relative to non-human primates. The p-value for this effect was {p_human:.3g}."
    )
    explanation_parts.append(
        f"For a typical specimen (median age, prob_male = 0.5, anterior teeth), the model-predicted AMTL "
        f"probability was {p_nonhuman * 100:.2f}% for non-human primates and {p_human_pred * 100:.2f}% "
        f"for humans, a difference of {(p_human_pred - p_nonhuman) * 100:.2f} percentage points."
    )

    if likert >= 60:
        answer_text = (
            "These results provide evidence that, after accounting for age, sex, and tooth class, "
            "modern humans have higher frequencies of antemortem tooth loss than the sampled non-human primate genera."
        )
    elif likert <= 40:
        answer_text = (
            "Overall, the regression results do not provide strong or consistent evidence that modern humans "
            "have higher AMTL frequencies than non-human primates once age, sex, and tooth class are controlled for."
        )
    else:
        answer_text = (
            "The regression results suggest only weak evidence that modern humans might have higher AMTL frequencies "
            "than non-human primates after adjusting for age, sex, and tooth class."
        )

    explanation_parts.append(answer_text)

    explanation = " ".join(explanation_parts)

    # Write JSON output to conclusion.txt
    output = {
        "response": int(likert),
        "explanation": explanation,
    }
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

