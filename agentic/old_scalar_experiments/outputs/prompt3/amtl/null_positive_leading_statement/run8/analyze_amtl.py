import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep only the genera relevant to the research question.
    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Basic cleaning and sanity checks.
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])

    if df.empty:
        conclusion = {
            "response": "No",
            "strength": 0,
            "confidence": 10,
            "explanation": (
                "After filtering to the relevant genera and required variables, "
                "the dataset contained no usable observations, so the comparison "
                "between modern humans and non-human primates could not be made."
            ),
        }
        Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))
        return

    # Response as a proportion with binomial family, weighted by number of sockets.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens')) "
        "+ age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Examine genus effects relative to Homo sapiens (reference level).
    nonhuman = ["Pan", "Papio", "Pongo"]
    genus_effects = []
    for g in nonhuman:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        if term in result.params.index:
            beta = float(result.params[term])
            pval = float(result.pvalues[term])
            # logit(p_g) - logit(p_human) = beta  =>  OR(human vs g) = exp(-beta)
            or_h_vs_g = float(np.exp(-beta))
            genus_effects.append(
                {
                    "genus": g,
                    "beta": beta,
                    "pvalue": pval,
                    "or_h_vs_genus": or_h_vs_g,
                }
            )

    if not genus_effects:
        conclusion = {
            "response": "No",
            "strength": 0,
            "confidence": 15,
            "explanation": (
                "The fitted model did not estimate separate effects for any non-human "
                "primate genera, so a direct comparison with modern humans was not "
                "possible from this dataset."
            ),
        }
        Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))
        return

    count_strong = sum(ge["beta"] < 0 and ge["pvalue"] < 0.05 for ge in genus_effects)
    count_weaker = sum(ge["beta"] < 0 and ge["pvalue"] >= 0.05 for ge in genus_effects)
    count_reversed = sum(ge["beta"] > 0 for ge in genus_effects)

    # Predicted probabilities at mean age/sex, averaged across tooth classes.
    avg_age = float(df["age"].mean())
    avg_prob_male = float(df["prob_male"].mean())
    tooth_classes = df["tooth_class"].unique()

    pred_probs: dict[str, float] = {}
    for g in ["Homo sapiens"] + nonhuman:
        if g not in df["genus"].unique():
            continue
        tmp = pd.DataFrame(
            {
                "age": [avg_age] * len(tooth_classes),
                "prob_male": [avg_prob_male] * len(tooth_classes),
                "genus": [g] * len(tooth_classes),
                "tooth_class": tooth_classes,
                "sockets": [1] * len(tooth_classes),
                "prop_amtl": [0.0] * len(tooth_classes),
            }
        )
        mu = result.predict(tmp)
        pred_probs[g] = float(mu.mean())

    human_prob = pred_probs.get("Homo sapiens", float("nan"))
    nonhuman_probs = [p for g, p in pred_probs.items() if g != "Homo sapiens"]
    max_nonhuman_prob = max(nonhuman_probs) if nonhuman_probs else float("nan")

    if (
        np.isfinite(human_prob)
        and np.isfinite(max_nonhuman_prob)
        and human_prob > max_nonhuman_prob
        and count_reversed == 0
    ):
        response = "Yes"
        base_strength = 60 + 10 * count_strong + 5 * count_weaker
        if max_nonhuman_prob > 0:
            rel_diff = (human_prob - max_nonhuman_prob) / max_nonhuman_prob
            base_strength += min(20.0, max(0.0, rel_diff * 100.0 / 5.0))
        strength = int(max(0, min(100, round(base_strength))))

        conf_base = 50 + 10 * count_strong + 5 * count_weaker
        conf_base += min(15.0, np.log10(len(df) + 1) * 5.0)
        confidence = int(max(0, min(100, round(conf_base))))
    else:
        response = "No"
        base_strength = 50 + 15 * count_reversed
        strength = int(max(0, min(100, round(base_strength))))
        conf_base = 40 + 10 * count_reversed
        conf_base += min(10.0, np.log10(len(df) + 1) * 3.0)
        confidence = int(max(0, min(100, round(conf_base))))

    lines: list[str] = []
    lines.append(
        "I fit a binomial logistic regression for the proportion of antemortem tooth "
        "loss (num_amtl / sockets) with genus, age at death, sex (prob_male), and "
        "tooth class as predictors, weighting each observation by the number of "
        "observable sockets."
    )
    lines.append(
        "Genus was coded with modern humans (Homo sapiens) as the reference level, so "
        "negative coefficients for other genera indicate lower odds of AMTL than in "
        "modern humans after adjusting for age, sex, and tooth class."
    )

    if genus_effects:
        parts = []
        for ge in genus_effects:
            direction = "lower" if ge["beta"] < 0 else "higher"
            parts.append(
                f"{ge['genus']} (coefficient {ge['beta']:.3f}, p-value {ge['pvalue']:.3g}, "
                f"odds ratio for humans vs this genus {ge['or_h_vs_genus']:.2f}, "
                f"meaning humans have {direction} log-odds of AMTL if the coefficient "
                f"is negative)."
            )
        lines.append(
            "The estimated effects for each non-human genus were: " + " ".join(parts)
        )

    if "Homo sapiens" in pred_probs and nonhuman_probs:
        nh_summary = ", ".join(
            f"{g}: {pred_probs[g]:.3f}"
            for g in pred_probs
            if g != "Homo sapiens"
        )
        lines.append(
            "At the mean age and sex and averaged across tooth classes, the predicted "
            f"probability that a socket shows AMTL was {pred_probs['Homo sapiens']:.3f} "
            f"for modern humans compared to {nh_summary} for the non-human genera."
        )

    if response == "Yes":
        lines.append(
            "Across the non-human genera represented in the data, modern humans show "
            "consistently higher modeled probabilities and odds of AMTL after adjusting "
            "for age, sex, and tooth class, providing support for the hypothesis."
        )
    else:
        lines.append(
            "After adjusting for age, sex, and tooth class, modern humans do not show "
            "a consistently higher modeled probability of AMTL than all non-human "
            "genera, so the hypothesis is not strongly supported by this analysis."
        )

    explanation = " ".join(lines)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

