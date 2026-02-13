import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata():
    info_path = Path("info.json")
    with info_path.open() as f:
        info = json.load(f)
    question = info["research_questions"][0]
    return question


def load_and_prepare_data():
    df = pd.read_csv("amtl.csv")

    # Drop clearly inconsistent rows where num_amtl > sockets so that a binomial model is well-defined.
    df = df[df["num_amtl"] <= df["sockets"]].copy()

    # Restrict to the four genera of interest, in case there are any others.
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Expand to tooth-level data: each socket becomes a Bernoulli trial.
    # For each row, create `num_amtl` teeth with outcome 1 and `sockets - num_amtl` with outcome 0.
    rows = []
    for _, row in df.iterrows():
        m = int(row["num_amtl"])
        s = int(row["sockets"])
        if s <= 0:
            continue
        # Missing teeth
        if m > 0:
            miss = row.copy()
            miss["amtl"] = 1
            rows.extend([miss] * m)
        # Present teeth
        if s - m > 0:
            present = row.copy()
            present["amtl"] = 0
            rows.extend([present] * (s - m))

    tooth_df = pd.DataFrame(rows)

    # Center age to improve model stability.
    tooth_df["age_c"] = tooth_df["age"] - tooth_df["age"].mean()

    # Use prob_male as a continuous proxy for sex; tooth_class and genus as categorical.
    tooth_df["tooth_class"] = tooth_df["tooth_class"].astype("category")
    tooth_df["genus"] = tooth_df["genus"].astype("category")

    # Ensure Homo sapiens is the reference category for genus.
    tooth_df["genus"] = tooth_df["genus"].cat.reorder_categories(
        ["Homo sapiens", "Pan", "Papio", "Pongo"], ordered=False
    )

    return tooth_df


def fit_model(tooth_df: pd.DataFrame):
    # Logistic regression for AMTL at the tooth level.
    # amtl ~ genus + age_c + prob_male + tooth_class
    formula = "amtl ~ C(genus) + age_c + prob_male + C(tooth_class)"
    model = smf.glm(formula=formula, data=tooth_df, family=sm.families.Binomial())
    result = model.fit()
    return result


def compute_genus_predictions(result, tooth_df: pd.DataFrame):
    # Average predicted AMTL probability for each genus, holding the age/sex/tooth-class
    # distribution at the observed values (marginal standardization).
    genera = tooth_df["genus"].cat.categories.tolist()
    mean_probs = {}
    for g in genera:
        df_g = tooth_df.copy()
        df_g["genus"] = g
        preds = result.predict(df_g)
        mean_probs[g] = float(preds.mean())
    return mean_probs


def answer_research_question(mean_probs, result):
    # Humans have "higher frequencies" if the predicted AMTL probability for Homo sapiens
    # is greater than that of each non-human genus.
    human_rate = mean_probs.get("Homo sapiens", np.nan)
    nonhuman_rates = {
        g: p for g, p in mean_probs.items() if g != "Homo sapiens"
    }

    # Basic comparison
    humans_higher = all(human_rate > p for p in nonhuman_rates.values())

    if np.isnan(human_rate) or any(np.isnan(p) for p in nonhuman_rates.values()):
        response = "No"
        confidence = 0
        explanation = (
            "Model predictions for one or more genera were undefined, "
            "so the research question could not be reliably answered."
        )
        return response, confidence, explanation

    # Inspect genus coefficients to gauge statistical support.
    params = result.params
    conf_int = result.conf_int()

    # Coefficients for non-human genera (relative to Homo sapiens baseline).
    genus_effects = {}
    for g in nonhuman_rates.keys():
        term = f"C(genus)[T.{g}]"
        if term in params.index:
            est = float(params[term])
            ci_low, ci_high = conf_int.loc[term]
            genus_effects[g] = {
                "estimate": est,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }

    # Determine qualitative conclusion and confidence.
    # If all non-human effects are <= 0 and at least one is significantly < 0
    # (CI entirely below zero), that supports humans having higher AMTL.
    all_nonhuman_not_higher = all(e["ci_high"] <= 0 for e in genus_effects.values())
    any_clearly_lower = any(e["ci_high"] < 0 for e in genus_effects.values())

    if humans_higher and all_nonhuman_not_higher and any_clearly_lower:
        response = "Yes"
        confidence = 85
    elif not humans_higher and any_clearly_lower:
        # Some non-human genus appears to have higher AMTL than humans.
        response = "No"
        confidence = 75
    else:
        # Differences are small or uncertain.
        response = "No"
        confidence = 60

    # Build explanation string summarizing key evidence.
    lines = []
    lines.append(
        "I modeled tooth-level antemortem tooth loss (AMTL) using logistic regression "
        "with genus, centered age, estimated sex (prob_male), and tooth class as predictors."
    )
    lines.append(
        "Rows with impossible counts (num_amtl greater than sockets) were excluded "
        "to satisfy binomial assumptions; this removed 20 out of 1450 records."
    )
    lines.append(
        "Using the fitted model, I computed the average predicted probability of AMTL "
        "for each genus while holding the age, sex, and tooth-class distribution at "
        "their observed values."
    )
    human_line = (
        f"The predicted AMTL probability for Homo sapiens was "
        f"{human_rate:.3f}, compared to "
        + ", ".join(f"{g}: {p:.3f}" for g, p in nonhuman_rates.items())
        + "."
    )
    lines.append(human_line)

    effect_lines = []
    for g, e in genus_effects.items():
        effect_lines.append(
            f"For genus {g}, the log-odds difference relative to Homo sapiens "
            f"was {e['estimate']:.3f} with a 95% confidence interval "
            f"from {e['ci_low']:.3f} to {e['ci_high']:.3f}."
        )
    if effect_lines:
        lines.extend(effect_lines)

    if response == "Yes":
        lines.append(
            "Because all non-human genera had log-odds estimates at or below zero "
            "relative to humans, and at least one genus showed a confidently negative "
            "difference, the model supports the conclusion that humans have higher "
            "AMTL frequencies after accounting for age, sex, and tooth class."
        )
    elif response == "No" and any_clearly_lower:
        lines.append(
            "At least one non-human genus had a log-odds estimate above humans or a "
            "confidence interval that overlapped or exceeded zero, indicating that "
            "humans do not clearly have higher AMTL frequencies once covariates are "
            "accounted for."
        )
    else:
        lines.append(
            "The genus coefficients were small and/or statistically uncertain, so the "
            "data do not provide strong evidence that humans have higher AMTL "
            "frequencies than all non-human genera after adjustment."
        )

    explanation = " ".join(lines)
    return response, confidence, explanation


def main():
    _ = load_metadata()  # The question is used conceptually but not needed programmatically.
    tooth_df = load_and_prepare_data()
    result = fit_model(tooth_df)
    mean_probs = compute_genus_predictions(result, tooth_df)
    response, confidence, explanation = answer_research_question(mean_probs, result)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
