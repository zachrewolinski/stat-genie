import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def build_explanation(
    response: str,
    n_total: int,
    genus_counts: Dict[str, int],
    genus_props: Dict[str, float],
    coef: float,
    pval: float,
    odds_ratio: float,
    mean_age: float,
    mean_prob_male: float,
    tooth_mode: str,
    prob_human: float,
    prob_nonhuman: float,
) -> str:
    genus_count_str = ", ".join(
        f"{g}: {count}" for g, count in sorted(genus_counts.items())
    )
    genus_prop_str = ", ".join(
        f"{g}: {prop:.3f}" for g, prop in sorted(genus_props.items())
    )

    base_description = (
        "I analyzed the antemortem tooth loss (AMTL) dataset by modelling the "
        "proportion of missing teeth (num_amtl out of sockets) using a binomial "
        "logistic regression. The main predictor was whether a specimen was a "
        "modern human (Homo sapiens) versus a non-human primate (Pan, Pongo, Papio), "
        "and I included age, the probability of being male (prob_male), and tooth "
        "class (anterior/posterior/premolar) as covariates to account for their "
        "effects on AMTL.\n"
        f"The model was fit to {n_total} observations after filtering the data to "
        "the four genera of interest. The number of observations per genus was "
        f"{genus_count_str}. The observed mean proportion of missing teeth "
        f"(num_amtl summed over sockets) by genus was {genus_prop_str}.\n"
        f"In the fitted model, the coefficient for the human indicator "
        f"(Homo sapiens vs. non-human primates) was {coef:.3f} on the log-odds "
        f"scale (odds ratio {odds_ratio:.2f}, p-value {pval:.3g}). To interpret "
        "this in more concrete terms, I computed predicted probabilities of a "
        "socket being missing for a typical individual with age "
        f"{mean_age:.1f} years, mean sex probability (prob_male={mean_prob_male:.2f}), "
        f"and the most common tooth class ('{tooth_mode}'). Under this scenario, "
        f"the predicted probability of AMTL at a socket was {prob_human:.3f} for "
        f"humans and {prob_nonhuman:.3f} for non-human primates.\n"
    )

    if response == "Yes":
        conclusion_sentence = (
            "Because the human indicator has a positive coefficient and the p-value "
            "is below a conventional 0.05 threshold, the model provides strong "
            "evidence that, after controlling for age, sex, and tooth class, modern "
            "humans have a higher frequency of antemortem tooth loss than the "
            "non-human primate genera in this sample."
        )
    else:
        conclusion_sentence = (
            "The estimated human effect is not both clearly positive and statistically "
            "significant at a conventional 0.05 level, so the model does not provide "
            "strong evidence that modern humans have higher AMTL frequencies than "
            "the non-human primate genera once age, sex, and tooth class are taken "
            "into account."
        )

    return base_description + conclusion_sentence


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep only genera relevant to the research question.
    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Drop rows with missing values in variables used in the model.
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )

    # Define human indicator and proportion of missing teeth.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Fit binomial logistic regression using proportions with socket counts as weights.
    model = smf.glm(
        "prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef = float(model.params["is_human"])
    pval = float(model.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Typical covariate values for prediction.
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    tooth_mode = str(df["tooth_class"].mode().iloc[0])

    pred_df = pd.DataFrame(
        {
            "prop_amtl": [0.0, 0.0],
            "is_human": [1, 0],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [tooth_mode, tooth_mode],
        }
    )
    probs = model.predict(pred_df)
    prob_human = float(probs.iloc[0])
    prob_nonhuman = float(probs.iloc[1])

    has_higher = (coef > 0.0) and (pval < 0.05)
    response = "Yes" if has_higher else "No"

    n_total = int(len(df))
    genus_counts = {str(k): int(v) for k, v in df["genus"].value_counts().items()}
    genus_props = {
        str(g): float(group["num_amtl"].sum() / group["sockets"].sum())
        for g, group in df.groupby("genus")
    }

    explanation = build_explanation(
        response=response,
        n_total=n_total,
        genus_counts=genus_counts,
        genus_props=genus_props,
        coef=coef,
        pval=pval,
        odds_ratio=odds_ratio,
        mean_age=mean_age,
        mean_prob_male=mean_prob_male,
        tooth_mode=tooth_mode,
        prob_human=prob_human,
        prob_nonhuman=prob_nonhuman,
    )

    result = {"response": response, "explanation": explanation}

    # Write the required JSON output; do not append extra text or lines.
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)

    # Print a brief summary to stdout for transparency.
    print("response:", response)
    print("is_human coef:", coef, "p-value:", pval, "odds_ratio:", odds_ratio)
    print("predicted AMTL prob human/non-human:", prob_human, prob_nonhuman)


if __name__ == "__main__":
    main()

