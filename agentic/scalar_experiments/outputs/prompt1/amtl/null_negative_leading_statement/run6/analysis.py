import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic sanity checks
    df = df.copy()
    df = df[df["sockets"] > 0]
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Ensure Homo sapiens is present and treat it as the reference genus
    if "Homo sapiens" not in set(df["genus"]):
        raise ValueError("Expected genus 'Homo sapiens' not found in data.")

    # Binomial regression: per-tooth probability of AMTL
    # We use grouped binomial with freq_weights = sockets to respect varying numbers of sockets.
    formula = "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    # Extract adjusted genus-level AMTL probabilities by averaging predictions
    genera = sorted(df["genus"].unique())
    base_covariates = df[["age", "prob_male", "tooth_class"]]

    adjusted_probs = {}
    for g in genera:
        df_new = base_covariates.copy()
        df_new["genus"] = g
        # Predict per-socket probability for each observation if it belonged to genus g
        p = model.predict(df_new)
        # Weight by number of sockets to approximate overall per-socket AMTL probability
        adjusted_probs[g] = float(np.average(p, weights=df["sockets"]))

    homo_key = "Homo sapiens"
    homo_prob = adjusted_probs[homo_key]
    nonhuman_probs = {g: p for g, p in adjusted_probs.items() if g != homo_key}

    # Determine answer: does Homo sapiens have higher AMTL than all non-human genera?
    max_nonhuman_prob = max(nonhuman_probs.values())
    humans_higher = homo_prob > max_nonhuman_prob

    # Also consider statistical evidence from genus coefficients
    params = model.params
    bse = model.bse

    genus_effects = {}
    for g in nonhuman_probs.keys():
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        if term in params:
            coef = float(params[term])
            se = float(bse[term])
            z = coef / se if se > 0 else np.nan
            genus_effects[g] = {"coef": coef, "se": se, "z": z}

    # We will only answer "Yes" if humans clearly have the highest adjusted probability
    # and all non-human genera show lower or equal AMTL (negative or near-zero coefficients).
    # Otherwise, we answer "No".
    response = "Yes" if humans_higher else "No"

    # Build explanation text
    lines = []
    lines.append("Research question: Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?")
    lines.append(f"We analysed {len(df)} specimen-tooth-class rows using a binomial regression model with a logit link, modelling the per-tooth probability of AMTL (num_amtl / sockets) with sockets as binomial trial counts.")
    lines.append("The model included genus, age, estimated sex (prob_male), and tooth class (anterior/posterior/premolar) as predictors, treating Homo sapiens as the reference genus.")
    lines.append("From the fitted model we obtained adjusted AMTL probabilities by predicting, for each observation, the per-tooth AMTL probability if it belonged to each genus while holding age, sex, and tooth class at their observed values, and then averaging predictions weighted by the number of sockets.")
    lines.append("Adjusted per-tooth AMTL probabilities by genus (higher values indicate more frequent tooth loss):")
    for g, p in adjusted_probs.items():
        lines.append(f"  - {g}: {p:.4f}")
    lines.append(f"For modern humans (Homo sapiens) the adjusted AMTL probability was {homo_prob:.4f}, while the highest adjusted probability among non-human genera was {max_nonhuman_prob:.4f}.")

    if genus_effects:
        lines.append("Genus coefficients (log-odds) from the regression, relative to Homo sapiens (negative values indicate lower AMTL than humans, positive values higher):")
        for g, stats in genus_effects.items():
            lines.append(
                f"  - {g}: coef = {stats['coef']:.3f}, SE = {stats['se']:.3f}, z = {stats['z']:.2f}"
            )

    if response == "Yes":
        lines.append("Because Homo sapiens shows a higher adjusted per-tooth AMTL probability than all non-human genera in this model, after adjusting for age, sex, and tooth class, we conclude that modern humans do have higher frequencies of AMTL compared to the non-human primate genera considered.")
    else:
        lines.append("Because Homo sapiens does not have a clearly higher adjusted per-tooth AMTL probability than all non-human genera in this model (and at least one non-human genus matches or exceeds the human AMTL rate), we conclude that modern humans do not have higher AMTL frequencies than the non-human primate genera considered, after adjusting for age, sex, and tooth class.")

    explanation = " ".join(lines)

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

