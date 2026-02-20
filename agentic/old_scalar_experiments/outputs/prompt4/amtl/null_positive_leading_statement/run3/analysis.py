import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Sanity check: ensure counts are within possible range
    df = df[(df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])].copy()

    # Proportion of missing teeth for modeling; use sockets as binomial weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Use Pan (chimpanzee) as the reference genus so that the human coefficient
    # directly reflects the contrast with a non-human primate.
    formula = "prop_amtl ~ C(genus, Treatment(reference='Pan')) + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Compute standardized predicted probabilities by genus:
    # for each genus, predict AMTL probability if it had the same
    # age/sex/tooth_class distribution as the full dataset.
    genera = sorted(df["genus"].unique())
    genus_pred = {}

    for g in genera:
        df_g = df.copy()
        df_g["genus"] = g
        pred = result.predict(df_g)
        # Average per-socket probability, weighting by the number of sockets
        mean_pred = float(np.average(pred, weights=df["sockets"]))
        genus_pred[g] = mean_pred

    # Extract key contrasts where available
    params = result.params
    pvalues = result.pvalues

    def coef_info(term: str):
        if term in params.index:
            return float(params[term]), float(pvalues[term])
        return None, None

    homo_vs_pan_coef, homo_vs_pan_p = coef_info("C(genus, Treatment(reference='Pan'))[T.Homo sapiens]")
    pongo_vs_pan_coef, pongo_vs_pan_p = coef_info("C(genus, Treatment(reference='Pan'))[T.Pongo]")
    papio_vs_pan_coef, papio_vs_pan_p = coef_info("C(genus, Treatment(reference='Pan'))[T.Papio]")

    # Determine answer: does Homo sapiens have higher AMTL than all non-human genera?
    homo_pred = genus_pred.get("Homo sapiens")
    non_human_preds = {g: p for g, p in genus_pred.items() if g != "Homo sapiens"}

    # Basic logical decision based on standardized predicted probabilities
    homo_higher_than_all = all(
        (homo_pred is not None)
        and (p is not None)
        and homo_pred > p
        for p in non_human_preds.values()
    )

    # Also check that the Homo-vs-Pan contrast is positive and statistically strong, if present
    strong_stat_evidence = False
    if homo_vs_pan_coef is not None and homo_vs_pan_p is not None:
        strong_stat_evidence = (homo_vs_pan_coef > 0.0) and (homo_vs_pan_p < 0.01)

    if homo_higher_than_all and strong_stat_evidence:
        response_value = 90
        yes_no_text = "Yes"
    elif homo_higher_than_all:
        response_value = 75
        yes_no_text = "Yes"
    else:
        # Either humans do not clearly have higher predicted AMTL
        # than all non-human genera, or statistical evidence is weak.
        response_value = 25
        yes_no_text = "No"

    # Build a concise but informative explanation
    lines = []
    lines.append(
        "I fitted a binomial regression model with the proportion of missing teeth "
        "(num_amtl / sockets) as the outcome, using a logit link and weighting by "
        "the number of observable sockets for each specimen."
    )
    lines.append(
        "The predictors in the model were genus (Pan as the reference category), "
        "age at death, estimated probability of being male, and tooth class "
        "(anterior, premolar, posterior), so the genus effects represent differences "
        "in AMTL frequencies after accounting for age, sex, and tooth class."
    )

    if homo_vs_pan_coef is not None and homo_vs_pan_p is not None:
        direction = "higher" if homo_vs_pan_coef > 0.0 else "lower"
        significance = (
            "statistically significant"
            if homo_vs_pan_p < 0.05
            else "not statistically significant"
        )
        lines.append(
            f"In this model, the coefficient comparing Homo sapiens to Pan on the "
            f"log-odds scale was {homo_vs_pan_coef:.3f} with p-value {homo_vs_pan_p:.3g}, "
            f"meaning that, after adjustment, humans have {direction} AMTL frequencies than chimpanzees "
            f"and this difference is {significance} at the 0.05 level."
        )

    # Summarize standardized predicted probabilities by genus
    if homo_pred is not None and non_human_preds:
        genus_summaries = []
        for g, p in genus_pred.items():
            genus_summaries.append(f"{g}: {p:.3f}")
        lines.append(
            "Using the fitted model, I computed standardized predicted probabilities of AMTL by "
            "setting each observation's genus in turn to each taxon while keeping age, sex, and "
            "tooth class at their observed values, then averaging the predicted probabilities "
            "weighted by sockets. The resulting standardized AMTL probabilities were: "
            + "; ".join(genus_summaries)
            + "."
        )

        # Describe how humans compare to each non-human genus in the standardized predictions.
        comparison_sentences = []
        for g, p in non_human_preds.items():
            if homo_pred > p:
                relation = "higher"
            elif homo_pred < p:
                relation = "lower"
            else:
                relation = "similar"
            comparison_sentences.append(
                f"Homo sapiens has {relation} standardized predicted AMTL probability than {g}."
            )
        lines.append(" ".join(comparison_sentences))

    lines.append(
        f"Based on these results, the answer to the question "
        f"\"Do modern humans have higher frequencies of antemortem tooth loss than non-human primates, "
        f"after accounting for age, sex, and tooth class?\" is \"{yes_no_text}\"."
    )

    explanation = " ".join(lines)

    conclusion = {"response": response_value, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
