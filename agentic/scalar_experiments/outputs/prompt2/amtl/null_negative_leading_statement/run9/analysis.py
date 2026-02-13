import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Drop any rows with missing critical fields
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])

    # Ensure valid binomial data
    df = df[df["sockets"] > 0].copy()
    df = df[(df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])].copy()

    # Proportion of antemortem tooth loss per specimen/tooth class
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Restrict to the genera of interest
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    if df.empty:
        raise RuntimeError("Filtered dataset is empty after restricting to target genera.")

    # Binomial GLM with Homo sapiens as reference genus.
    # We model the AMTL proportion with number of sockets as binomial trial weights,
    # adjusting for age, sex (prob_male), and tooth class.
    formula = "amtl_prop ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract coefficients for non-human genera relative to Homo sapiens
    params = result.params
    conf_int = result.conf_int()

    genus_effects = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term not in params.index:
            continue
        est = float(params[term])
        ci_low, ci_high = conf_int.loc[term].astype(float)
        genus_effects[genus] = {
            "coef": est,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    # Compute adjusted predicted probabilities at representative covariate values
    # (mean age, mean sex probability, and most common tooth class).
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    common_tooth_class = df["tooth_class"].mode().iloc[0]

    pred_rows = []
    for genus in genera_of_interest:
        pred_rows.append(
            {
                "genus": genus,
                "age": mean_age,
                "prob_male": mean_prob_male,
                "tooth_class": common_tooth_class,
                "amtl_prop": 0.0,
                "sockets": 1.0,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    preds = result.predict(pred_df)
    pred_probs = dict(zip(pred_df["genus"], preds))

    # Determine answer: do modern humans have higher AMTL frequencies?
    human_prob = float(pred_probs.get("Homo sapiens", np.nan))
    nonhuman_probs = [float(pred_probs[g]) for g in ["Pan", "Papio", "Pongo"] if g in pred_probs]

    # Basic decision rule: compare adjusted predicted probabilities and
    # consider the sign and uncertainty of genus coefficients.
    humans_higher = all(human_prob > p for p in nonhuman_probs)

    # Also incorporate coefficient directions: if all non-human coefficients are
    # clearly negative (CI entirely below 0), that supports humans having higher AMTL.
    all_nonhuman_ci_below_zero = bool(
        genus_effects
        and all(info["ci_high"] < 0.0 for info in genus_effects.values())
    )

    if humans_higher and all_nonhuman_ci_below_zero:
        response = "Yes"
        confidence = 85
    elif humans_higher:
        response = "Yes"
        confidence = 65
    else:
        response = "No"
        # If at least one non-human coefficient is clearly above zero, we are quite confident
        any_nonhuman_ci_above_zero = any(info["ci_low"] > 0.0 for info in genus_effects.values())
        confidence = 85 if any_nonhuman_ci_above_zero else 65

    # Build explanation summarizing key evidence
    coef_summaries = []
    for genus, info in genus_effects.items():
        coef_summaries.append(
            f"{genus}: coef={info['coef']:.3f}, 95% CI=[{info['ci_low']:.3f}, {info['ci_high']:.3f}]"
        )
    coef_text = "; ".join(coef_summaries)

    prob_summaries = []
    for genus in genera_of_interest:
        if genus in pred_probs:
            prob_summaries.append(f"{genus}: predicted AMTL proportion≈{pred_probs[genus]:.3f}")
    prob_text = "; ".join(prob_summaries)

    explanation = (
        "I fit a binomial logistic regression model for the proportion of antemortem tooth loss "
        "(num_amtl / sockets) with genus, age, sex (prob_male), and tooth class as predictors, "
        "treating Homo sapiens as the reference genus and weighting each observation by the number "
        "of observable sockets. The coefficients for the non-human genera represent their AMTL "
        "log-odds relative to modern humans after accounting for age, sex, and tooth class. "
        f"Genus coefficients relative to Homo sapiens were: {coef_text}. "
        f"Adjusted predicted AMTL proportions at mean age, mean sex, and the most common tooth class were: {prob_text}. "
        "Based on these adjusted effects and their uncertainty, I evaluated whether modern humans have higher AMTL "
        "frequencies than Pan, Papio, and Pongo after controlling for the covariates."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

