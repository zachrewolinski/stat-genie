import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Construct binomial response: proportion of missing teeth with number of trials as sockets
    df = df.copy()
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Fit binomial regression with logit link
    formula = (
        'amtl_rate ~ C(genus, Treatment(reference="Homo sapiens"))'
        " + age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Standardized predicted AMTL rates for each genus:
    # replicate the dataset but set genus to each level in turn
    genera = sorted(df["genus"].unique())
    standardized_preds = {}
    for g in genera:
        df_g = df.copy()
        df_g["genus"] = g
        standardized_preds[g] = float(result.predict(df_g).mean())

    # Extract genus coefficients (non‑human vs Homo sapiens)
    target_genera = ["Pan", "Papio", "Pongo"]
    params = result.params
    pvalues = result.pvalues

    evidence_score = 0
    genus_details = []
    for g in target_genera:
        param_name = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
        if param_name not in params:
            genus_details.append(
                f"{g}: no data/parameter estimated; treated as inconclusive."
            )
            continue
        beta = params[param_name]
        pval = pvalues[param_name]

        if beta < 0 and pval < 0.05:
            evidence_score += 2
            strength = "strong"
        elif beta < 0:
            evidence_score += 1
            strength = "weak"
        elif beta > 0 and pval < 0.05:
            evidence_score -= 2
            strength = "strong"
        else:
            evidence_score -= 1
            strength = "weak"

        genus_details.append(
            f"{g}: coefficient={beta:.3f}, p={pval:.3g} ({strength} evidence "
            f"{'lower' if beta < 0 else 'higher'} AMTL than humans)."
        )

    # Decide on overall answer based on evidence across genera
    # Also check standardized predicted AMTL rates
    human_rate = standardized_preds.get("Homo sapiens", np.nan)
    nonhuman_rates = [
        standardized_preds[g]
        for g in target_genera
        if g in standardized_preds
    ]
    humans_higher_pred = all(
        np.isfinite(human_rate) and human_rate > r for r in nonhuman_rates
    ) if nonhuman_rates else False

    if evidence_score > 0 and humans_higher_pred:
        response = "Yes"
    else:
        response = "No"

    # Map evidence_score and prediction separation to a 0–100 confidence
    max_score = 6.0
    min_score = -6.0
    clipped = max(min(evidence_score, max_score), min_score)
    score_component = (clipped + max_score) / (2 * max_score)  # 0–1

    if nonhuman_rates and np.isfinite(human_rate):
        mean_gap = float(
            human_rate - float(np.mean(nonhuman_rates))
        )
        gap_component = max(min(mean_gap * 50.0, 1.0), -1.0)
        gap_component = (gap_component + 1.0) / 2.0
    else:
        gap_component = 0.5

    raw_conf = 100.0 * (0.6 * score_component + 0.4 * gap_component)
    confidence = int(round(max(0.0, min(100.0, raw_conf))))

    # Build explanation string
    preds_str = ", ".join(
        f"{g}: {standardized_preds[g]:.3f}" for g in genera
    )
    explanation_parts = [
        "I fit a binomial regression model (logit link) with the proportion "
        "of missing teeth (num_amtl/sockets) as the response, and genus, age, "
        "sex (prob_male), and tooth_class as predictors, weighting each row by "
        "the number of observable sockets.",
        "Humans (Homo sapiens) were used as the reference genus so negative "
        "coefficients for the Pan, Papio, and Pongo indicators indicate lower "
        "AMTL than in humans after accounting for age, sex, and tooth class.",
        "Standardized predicted AMTL rates (applying each genus label to the "
        "full covariate distribution) were: " + preds_str + ".",
        "Genus-specific effects were: " + " ".join(genus_details),
        f"Based on these results, the evidence_score={evidence_score} and "
        f"humans_higher_pred={humans_higher_pred}, which supports the overall "
        f"answer of '{response}' to the question of whether modern humans have "
        "higher AMTL frequencies than the non-human primate genera considered.",
    ]
    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

