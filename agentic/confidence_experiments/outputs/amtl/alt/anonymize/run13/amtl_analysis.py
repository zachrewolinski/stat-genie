import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial (logit) regression for AMTL.

    The response is the proportion of missing teeth for a given
    specimen and tooth class, with the number of observable sockets
    used as binomial denominators via frequency weights.
    """
    df = df.copy()
    df["missing"] = df["feature3"].astype(float)
    df["sockets"] = df["feature4"].astype(float)
    df["prop_missing"] = df["missing"] / df["sockets"]

    # Ensure genus and tooth class are treated as categorical.
    # Homo sapiens will be the reference level for genus by default
    # (alphabetically first among the four genera present).
    formula = "prop_missing ~ C(feature8) + feature5 + feature7 + C(feature1)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_human_vs_nonhuman(result, df: pd.DataFrame):
    """
    Use the fitted model to compare predicted AMTL for
    Homo sapiens vs. each non-human genus, holding age,
    sex, and tooth class constant at representative values.
    """
    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]

    age_rep = float(df["feature5"].median())
    sex_rep = float(df["feature7"].median())
    tooth_rep = "Posterior"

    new_data = []
    for g in genera:
        new_data.append(
            {
                "feature8": g,
                "feature5": age_rep,
                "feature7": sex_rep,
                "feature1": tooth_rep,
            }
        )

    new_df = pd.DataFrame(new_data)
    preds = result.predict(new_df)

    predicted = dict(zip(genera, preds))

    # Extract genus coefficients and p-values to assess significance.
    params = result.params
    pvalues = result.pvalues

    genus_effects = {}
    for g in ["Pan", "Papio", "Pongo"]:
        term = f"C(feature8)[T.{g}]"
        if term in params:
            genus_effects[g] = {
                "coef": float(params[term]),
                "pvalue": float(pvalues[term]),
            }

    return predicted, genus_effects


def derive_likert_response(predicted, genus_effects):
    """
    Translate model predictions and significance into a 0-100 Likert score.

    0  = strong 'No' (clear evidence humans do NOT have higher AMTL)
    50 = indeterminate / no clear difference
    100 = strong 'Yes' (clear evidence humans have higher AMTL)
    """
    human_rate = predicted["Homo sapiens"]
    deltas = {}
    for g in ["Pan", "Papio", "Pongo"]:
        if g in predicted:
            deltas[g] = human_rate - predicted[g]

    # Determine directional consistency and significance.
    all_deltas_positive = all(d > 0 for d in deltas.values())
    any_delta_positive = any(d > 0 for d in deltas.values())

    # Significance: require p < 0.05 for a genus effect to be called significant.
    sig_flags = {}
    for g, eff in genus_effects.items():
        sig_flags[g] = eff["pvalue"] < 0.05

    num_sig_positive = sum(
        1
        for g, d in deltas.items()
        if d > 0 and sig_flags.get(g, False)
    )
    num_sig_total = sum(1 for g in ["Pan", "Papio", "Pongo"] if sig_flags.get(g, False))

    mean_delta = float(np.mean(list(deltas.values()))) if deltas else 0.0

    # Map to a Likert score.
    # Start at 50 (no evidence either way) and move up or down
    # based on direction, consistency, effect size, and significance.
    score = 50

    if all_deltas_positive and num_sig_positive == 3:
        # Consistent and significant higher AMTL for humans.
        if mean_delta >= 0.10:
            score = 90
        elif mean_delta >= 0.05:
            score = 80
        else:
            score = 70
    elif any_delta_positive and num_sig_positive >= 1:
        # Some evidence that humans have higher AMTL, but not uniformly.
        if mean_delta >= 0.05:
            score = 70
        else:
            score = 60
    elif not any_delta_positive and num_sig_total >= 1:
        # Significant evidence that humans are *not* higher.
        if mean_delta <= -0.10:
            score = 10
        elif mean_delta <= -0.05:
            score = 20
        else:
            score = 30
    else:
        # Inconclusive / weak evidence.
        if all_deltas_positive:
            score = 60
        elif not any_delta_positive:
            score = 40
        else:
            score = 50

    # Clip to [0, 100] and ensure integer.
    score = int(max(0, min(100, round(score))))

    return score, deltas, mean_delta, sig_flags


def build_explanation(
    score,
    predicted,
    deltas,
    mean_delta,
    genus_effects,
    sig_flags,
):
    lines = []
    lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher "
        "frequencies of antemortem tooth loss (AMTL) than non-human primates "
        "(Pan, Pongo, Papio), after accounting for age, sex, and tooth class?"
    )
    lines.append(
        "Method: I modeled the proportion of missing teeth (number of missing "
        "teeth divided by the number of observable sockets) using a binomial "
        "(logit-link) regression with predictors for genus, age at death, "
        "estimated sex, and tooth class (anterior/posterior/premolar). "
        "Each row in the dataset contributes a binomial observation with "
        "the number of sockets as the denominator."
    )

    lines.append(
        "To interpret the genus effect, I computed predicted AMTL frequencies "
        "for a representative individual (age and sex set to their sample medians, "
        "tooth class set to Posterior) for each genus."
    )

    # Predicted rates by genus
    for g, p in predicted.items():
        lines.append(f"Predicted AMTL frequency for {g}: {p:.3f}")

    # Differences relative to humans
    for g, d in deltas.items():
        direction = "higher" if d > 0 else "lower"
        lines.append(
            f"Compared to {g}, Homo sapiens have an estimated AMTL frequency "
            f"{abs(d):.3f} {direction} at the representative covariate values."
        )

    # Significance of genus coefficients
    for g, eff in genus_effects.items():
        direction = "lower" if eff["coef"] < 0 else "higher"
        sig_text = "statistically significant" if sig_flags.get(g, False) else "not statistically significant"
        lines.append(
            f"In the regression model, the coefficient for genus {g} "
            f"(relative to Homo sapiens) is {eff['coef']:.3f}, which implies "
            f"{direction} log-odds of AMTL for {g} compared with humans; "
            f"this effect is {sig_text} (p = {eff['pvalue']:.3f})."
        )

    if score >= 70:
        conclusion_phrase = (
            "Overall, these results provide strong evidence that modern humans "
            "have higher frequencies of antemortem tooth loss than the sampled "
            "non-human primate genera after accounting for age, sex, and tooth class."
        )
    elif score >= 60:
        conclusion_phrase = (
            "Overall, the evidence suggests that modern humans have somewhat "
            "higher AMTL frequencies than the sampled non-human primates, but "
            "the pattern is not uniformly strong or statistically significant "
            "across all genera."
        )
    elif score <= 30:
        conclusion_phrase = (
            "Overall, the model indicates that modern humans do not have higher "
            "AMTL frequencies than the sampled non-human primates once age, sex, "
            "and tooth class are taken into account; if anything, humans tend to "
            "show equal or lower AMTL frequencies."
        )
    else:
        conclusion_phrase = (
            "Overall, the evidence is mixed and does not clearly show that modern "
            "humans have higher AMTL frequencies than the sampled non-human primates "
            "after adjusting for age, sex, and tooth class."
        )

    lines.append(
        f"Mapping this evidence to a 0–100 Likert scale, where 0 represents a "
        f"strong 'No' and 100 a strong 'Yes' answer to the research question, "
        f"I assign a score of {score}."
    )
    lines.append(conclusion_phrase)

    explanation = "\n".join(lines)
    return explanation


def main():
    csv_path = Path("amtl.csv")
    if not csv_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory.")

    df = pd.read_csv(csv_path)

    result = fit_binomial_model(df)
    predicted, genus_effects = summarize_human_vs_nonhuman(result, df)
    score, deltas, mean_delta, sig_flags = derive_likert_response(
        predicted, genus_effects
    )
    explanation = build_explanation(
        score,
        predicted,
        deltas,
        mean_delta,
        genus_effects,
        sig_flags,
    )

    conclusion = {
        "response": int(score),
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

