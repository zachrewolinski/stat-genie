import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Compute proportion of antemortem tooth loss per tooth socket.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Ensure categorical coding with Homo sapiens as the reference genus.
    genus_order = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df["genus"] = pd.Categorical(df["genus"], categories=genus_order)
    df["tooth_class"] = pd.Categorical(df["tooth_class"])

    # Binomial regression with logit link; weights are number of sockets.
    model = smf.glm(
        formula="prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract genus coefficients (non-human genera vs Homo sapiens baseline).
    params = result.params
    pvalues = result.pvalues

    genus_effects = {}
    for genus in genus_order[1:]:
        term = f"C(genus)[T.{genus}]"
        if term in params:
            genus_effects[genus] = {
                "coef": float(params[term]),
                "pvalue": float(pvalues[term]),
            }

    # Marginal predicted AMTL probabilities per genus, averaging over the sample
    # and weighting by sockets.
    mean_pred = {}
    base_df = df.copy()
    for genus in genus_order:
        tmp = base_df.copy()
        tmp["genus"] = genus
        preds = result.predict(tmp)
        # Weight by number of sockets so each tooth socket contributes equally.
        mean_pred[genus] = float(np.average(preds, weights=base_df["sockets"]))

    # Compare Homo sapiens to non-human genera.
    human_rate = mean_pred["Homo sapiens"]
    nonhuman_rates = [mean_pred[g] for g in genus_order[1:]]
    avg_nonhuman = float(np.mean(nonhuman_rates))
    diff = human_rate - avg_nonhuman

    # Evidence strength: number of significantly lower non-human genera
    # (negative coefficient relative to Homo sapiens, p < 0.05).
    significant_lower = 0
    for genus, eff in genus_effects.items():
        if eff["coef"] < 0 and eff["pvalue"] < 0.05:
            significant_lower += 1

    # Map evidence and effect size to a 0–100 Likert-scale response.
    if diff > 0 and significant_lower == 3:
        if diff >= 0.05:
            response = 95
        elif diff >= 0.03:
            response = 90
        else:
            response = 80
    elif diff > 0 and significant_lower >= 1:
        response = 70
    elif diff > 0:
        response = 60
    elif abs(diff) <= 0.005:
        response = 50
    else:
        # Evidence that humans do not have higher AMTL.
        if diff < 0 and significant_lower == 3:
            response = 5
        else:
            response = 20

    # Build explanation text summarizing the analysis and key statistics.
    explanation_lines = []
    explanation_lines.append(
        "I modeled the proportion of antemortem tooth loss (num_amtl / sockets) "
        "using a binomial regression with a logit link, with predictors genus "
        "(Homo sapiens, Pan, Pongo, Papio), age-at-death, estimated probability "
        "of being male, and tooth class (anterior, posterior, premolar), and "
        "weighted each row by the number of observable tooth sockets."
    )

    explanation_lines.append(
        "In this model, Homo sapiens was treated as the reference genus, so the "
        "coefficients for the non-human genera represent differences in log-odds "
        "of AMTL relative to modern humans after adjusting for age, sex, and tooth class."
    )

    for genus in genus_order[1:]:
        eff = genus_effects.get(genus)
        if eff is not None:
            explanation_lines.append(
                f"For {genus}, the estimated coefficient relative to Homo sapiens "
                f"was {eff['coef']:.3f} with a p-value of {eff['pvalue']:.3g}."
            )

    explanation_lines.append(
        "To summarize the adjusted frequencies, I computed marginal predicted "
        "AMTL probabilities for each genus by setting genus in turn to each level "
        "while keeping the observed distributions of age, sex, and tooth class, "
        "and averaging the model-predicted probabilities across specimens "
        "weighted by the number of sockets."
    )

    explanation_lines.append(
        "The resulting estimated AMTL probabilities per tooth socket were: "
        + ", ".join(
            f"{genus}: {mean_pred[genus]:.3f}" for genus in genus_order
        )
        + "."
    )

    explanation_lines.append(
        f"On this adjusted scale, Homo sapiens had an estimated AMTL frequency "
        f"of {human_rate:.3f}, compared with an average of {avg_nonhuman:.3f} "
        f"for the three non-human genera (difference {diff:.3f})."
    )

    if diff > 0:
        if significant_lower == 3:
            explanation_lines.append(
                "All three non-human genera showed lower AMTL than Homo sapiens "
                "with statistically significant negative coefficients (p < 0.05), "
                "providing strong evidence that modern humans have higher AMTL "
                "frequencies after accounting for age, sex, and tooth class."
            )
        elif significant_lower >= 1:
            explanation_lines.append(
                "At least one non-human genus showed significantly lower AMTL "
                "than Homo sapiens, and the overall adjusted AMTL level was higher "
                "in Homo sapiens than in non-human primates, supporting the "
                "interpretation that humans tend to have higher AMTL frequencies."
            )
        else:
            explanation_lines.append(
                "Although the adjusted AMTL frequency was numerically higher in "
                "Homo sapiens, the genus coefficients were not consistently "
                "statistically significant, so the evidence that humans have "
                "higher AMTL frequencies is suggestive but not definitive."
            )
    elif abs(diff) <= 0.005:
        explanation_lines.append(
        "The adjusted AMTL frequencies for Homo sapiens and non-human genera "
        "were almost identical, and the genus coefficients showed little "
        "evidence of systematic differences, so the data do not support a clear "
        "difference in AMTL frequencies between humans and non-human primates "
        "once age, sex, and tooth class are controlled."
        )
    else:
        explanation_lines.append(
            "In this model, the adjusted AMTL frequency for Homo sapiens was "
            "lower than for the non-human genera on average, and/or the genus "
            "coefficients did not consistently favor higher AMTL in humans, so "
            "the data do not support the claim that humans have higher AMTL "
            "frequencies after accounting for age, sex, and tooth class."
        )

    explanation_lines.append(
        "These conclusions are based on a binomial regression that does not "
        "explicitly model specimen-level or population-level random effects, "
        "so while the evidence for or against higher human AMTL is grounded in "
        "statistical significance and effect sizes from this model, they should "
        "be interpreted with that limitation in mind."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(response), "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

