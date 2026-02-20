import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_FILE = Path("amtl.csv")
CONCLUSION_FILE = Path("conclusion.txt")


def fit_binomial_model(df: pd.DataFrame):
    df = df.copy()
    df = df[df["sockets"] > 0]
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    df["genus"] = pd.Categorical(df["genus"])
    if "Homo sapiens" in df["genus"].cat.categories:
        categories = list(df["genus"].cat.categories)
        categories.sort(key=lambda x: (x != "Homo sapiens", x))
        df["genus"] = df["genus"].cat.reorder_categories(categories)

    formula = "prop_amtl ~ genus + age + prob_male + tooth_class"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return df, result


def compute_genus_predictions(df: pd.DataFrame, result) -> dict:
    genera = list(df["genus"].cat.categories)

    def mean_pred_for_genus(genus_name: str) -> float:
        tmp = df.copy()
        tmp["genus"] = genus_name
        return float(result.predict(tmp).mean())

    return {g: mean_pred_for_genus(g) for g in genera}


def summarize_genus_effects(result) -> dict:
    params = result.params
    pvalues = result.pvalues
    summary = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"genus[T.{genus}]"
        if term in params.index:
            summary[genus] = {
                "coef": float(params[term]),
                "pvalue": float(pvalues[term]),
            }
        else:
            summary[genus] = None
    return summary


def derive_conclusion(predictions: dict, genus_effects: dict) -> dict:
    homo_key = "Homo sapiens"
    homo_pred = predictions.get(homo_key)
    other_genera = {g: p for g, p in predictions.items() if g != homo_key}

    if homo_pred is None or not other_genera:
        response = "No"
        strength = 30
        confidence = 40
        explanation = (
            "The model could not clearly estimate AMTL frequencies for humans "
            "relative to non-human primates, so I cannot support a strong 'Yes' answer."
        )
        return {
            "response": response,
            "strength": strength,
            "confidence": confidence,
            "explanation": explanation,
        }

    max_other = max(other_genera.values())
    min_other = min(other_genera.values())
    effect_margin = float(homo_pred - max_other)
    yes_higher = homo_pred > max_other

    negative_and_sig = 0
    negative = 0
    for genus, stats in genus_effects.items():
        if stats is None:
            continue
        if stats["coef"] < 0:
            negative += 1
            if stats["pvalue"] < 0.05:
                negative_and_sig += 1

    if yes_higher and negative >= 2:
        response = "Yes"
    else:
        response = "No"

    margin_clipped = float(np.clip(effect_margin, -0.2, 0.2))
    margin_scale = abs(margin_clipped) / 0.2

    if response == "Yes":
        base_strength = 40 + 40 * margin_scale
        base_conf = 40 + 40 * margin_scale
    else:
        base_strength = 30 + 30 * margin_scale
        base_conf = 30 + 30 * margin_scale

    base_strength += 10 * min(negative_and_sig, 2)
    base_conf += 10 * min(negative_and_sig, 2)

    strength = int(round(max(0, min(100, base_strength))))
    confidence = int(round(max(0, min(100, base_conf))))

    explanation_lines = [
        "I fitted a binomial regression model for the proportion of antemortem tooth loss (AMTL) per tooth class,",
        "using sockets as binomial denominators and including genus, age at death, estimated sex (probability of male),",
        "and tooth class as predictors.",
        f"The model predicts an average AMTL frequency of approximately {homo_pred:.3f} for modern humans (Homo sapiens),",
        "compared to the following predicted frequencies for non-human primate genera:",
    ]
    for genus, p in other_genera.items():
        explanation_lines.append(f"- {genus}: {p:.3f}")

    explanation_lines.append(
        "Coefficients for Pan, Papio, and Pongo in the regression are the differences in log-odds of AMTL relative to humans."
    )
    for genus, stats in genus_effects.items():
        if stats is None:
            continue
        explanation_lines.append(
            f"For {genus}, the coefficient is {stats['coef']:.3f} with p-value {stats['pvalue']:.3g}."
        )

    if response == "Yes":
        explanation_lines.append(
            "Humans show higher predicted AMTL frequencies than all three non-human genera after adjusting for age, sex, and tooth class,"
        )
        explanation_lines.append(
            "and the negative genus coefficients (relative to humans) provide statistical support for this difference."
        )
    else:
        explanation_lines.append(
            "The model does not show humans having consistently higher AMTL than all non-human genera, "
            "or the estimated differences are small and/or statistically weak."
        )

    explanation = " ".join(explanation_lines)

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main():
    df = pd.read_csv(DATA_FILE)
    df, result = fit_binomial_model(df)
    predictions = compute_genus_predictions(df, result)
    genus_effects = summarize_genus_effects(result)
    conclusion = derive_conclusion(predictions, genus_effects)

    with CONCLUSION_FILE.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

