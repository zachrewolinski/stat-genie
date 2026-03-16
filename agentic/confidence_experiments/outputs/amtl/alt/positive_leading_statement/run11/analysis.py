import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Restrict to the four genera of interest
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Compute AMTL proportion per specimen / tooth class
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Basic descriptive stats: unadjusted AMTL frequency by genus
    genus_summary = (
        df.groupby("genus")["amtl_prop"].agg(["mean", "count"]).sort_index()
    )

    # Binomial (logistic) regression on the proportion with binomial family
    # and sockets as the number of trials (freq_weights).
    # Homo sapiens is set as the reference genus explicitly.
    formula = (
        "amtl_prop ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Adjusted predicted AMTL probabilities for each genus:
    # For each genus, set genus field while keeping age, prob_male, and tooth_class
    # as observed, then average predicted probabilities.
    def mean_pred_for_genus(genus: str) -> float:
        df_pred = df.copy()
        df_pred["genus"] = genus
        preds = result.predict(df_pred)
        return float(np.mean(preds))

    pred_means = {g: mean_pred_for_genus(g) for g in genera_of_interest}

    human_rate = pred_means["Homo sapiens"]
    nonhuman_genera = [g for g in genera_of_interest if g != "Homo sapiens"]
    nonhuman_rates = [pred_means[g] for g in nonhuman_genera]
    avg_nonhuman_rate = float(np.mean(nonhuman_rates))

    # Collect coefficient signs and p-values for genus contrasts vs Homo sapiens.
    genus_effects = {}
    for g in nonhuman_genera:
        coef_name = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        if coef_name in result.params:
            coef = float(result.params[coef_name])
            pval = float(result.pvalues[coef_name])
            # Negative coefficient => genus has lower log-odds of AMTL than humans.
            supports_humans_higher = coef < 0.0
            significant = pval < 0.05
            genus_effects[g] = {
                "coef": coef,
                "pval": pval,
                "supports_humans_higher": supports_humans_higher,
                "significant": significant,
                "pred_rate": pred_means[g],
            }

    # Determine overall strength of evidence that humans have higher AMTL frequency.
    n_support = sum(
        1 for g in genus_effects.values() if g["supports_humans_higher"]
    )
    n_sig_support = sum(
        1
        for g in genus_effects.values()
        if g["supports_humans_higher"] and g["significant"]
    )

    # Effect size: difference in adjusted AMTL proportion between humans
    # and the average of non-human genera.
    diff_human_vs_nonhuman = human_rate - avg_nonhuman_rate
    ratio_human_vs_nonhuman = (
        human_rate / avg_nonhuman_rate if avg_nonhuman_rate > 0 else np.nan
    )

    # Map evidence strength to a 0–100 Likert scale.
    # This is heuristic but grounded in the pattern of significance and effect size.
    if n_sig_support == len(nonhuman_genera):
        # Humans significantly higher than all non-human genera.
        if ratio_human_vs_nonhuman >= 2.0:
            response = 95
        elif ratio_human_vs_nonhuman >= 1.5:
            response = 90
        else:
            response = 85
    elif n_sig_support >= 2 and n_support == len(nonhuman_genera):
        response = 80
    elif n_sig_support >= 1 and n_support >= 2:
        response = 70
    elif n_support >= 2:
        response = 60
    elif n_support >= 1:
        response = 55
    else:
        # Little to no evidence that humans have higher AMTL.
        if diff_human_vs_nonhuman > 0:
            response = 50
        else:
            response = 40

    # Build a concise, human-readable explanation string.
    lines = []
    lines.append(
        "Research question: Do modern humans (Homo sapiens) show higher "
        "frequencies of antemortem tooth loss (AMTL) than non-human primate "
        "genera (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?"
    )

    # Unadjusted descriptive statistics
    desc_parts = []
    for genus, row in genus_summary.iterrows():
        desc_parts.append(
            f"{genus}: mean AMTL proportion ≈ {row['mean']:.3f} "
            f"across {int(row['count'])} specimen–tooth-class observations"
        )
    lines.append(
        "Unadjusted AMTL proportions by genus (missing teeth / observable "
        "sockets) show the following means: "
        + "; ".join(desc_parts)
        + "."
    )

    # Model description
    lines.append(
        "I fit a binomial (logistic) regression to the AMTL proportion per "
        "specimen–tooth-class, using the number of observable sockets as the "
        "binomial denominator (via frequency weights). The model included "
        "genus (Homo sapiens as the reference), age at death, estimated sex "
        "(probability of male), and tooth class (anterior, premolar, posterior)."
    )

    # Adjusted predicted probabilities
    pred_parts = []
    for genus in genera_of_interest:
        pred_parts.append(
            f"{genus}: adjusted AMTL probability ≈ {pred_means[genus]:.3f}"
        )
    lines.append(
        "Using this model, I computed adjusted AMTL probabilities for each "
        "genus by substituting each genus into the model while holding the "
        "age, sex, and tooth-class distributions at their observed values. "
        "The average predicted AMTL probabilities were: "
        + "; ".join(pred_parts)
        + "."
    )

    lines.append(
        f"On average, the adjusted AMTL probability for Homo sapiens "
        f"(≈ {human_rate:.3f}) was higher than the mean of the non-human "
        f"genera (≈ {avg_nonhuman_rate:.3f}), a difference of "
        f"≈ {diff_human_vs_nonhuman:.3f} in absolute probability and a "
        f"ratio of ≈ {ratio_human_vs_nonhuman:.2f}."
    )

    # Genus-specific inference
    genus_inference_parts = []
    for genus in nonhuman_genera:
        ge = genus_effects.get(genus)
        if ge is None:
            continue
        direction = "lower" if ge["supports_humans_higher"] else "higher"
        sig_phrase = (
            "statistically significant"
            if ge["significant"]
            else "not statistically significant at the 0.05 level"
        )
        genus_inference_parts.append(
            f"Relative to Homo sapiens, {genus} had {direction} adjusted "
            f"log-odds of AMTL (coefficient ≈ {ge['coef']:.3f}, "
            f"p ≈ {ge['pval']:.3f}), which is {sig_phrase}."
        )
    if genus_inference_parts:
        lines.append(
            "Genus-specific contrasts from the regression showed: "
            + " ".join(genus_inference_parts)
        )

    # Overall conclusion
    if response >= 75:
        qualitative = (
            "strong evidence that, after adjusting for age, sex, and tooth "
            "class, modern humans experience higher frequencies of AMTL than "
            "the non-human primate genera examined."
        )
    elif response >= 60:
        qualitative = (
            "moderate evidence that humans have higher AMTL frequencies than "
            "non-human primates after adjustment, although the strength of "
            "the effect varies across genera."
        )
    elif response >= 50:
        qualitative = (
            "suggestive but weak evidence that humans might have higher AMTL "
            "frequencies than non-human primates after adjustment; results "
            "are not consistently statistically significant."
        )
    else:
        qualitative = (
            "little to no reliable evidence that humans have higher AMTL "
            "frequencies than non-human primates once age, sex, and tooth "
            "class are controlled for."
        )

    lines.append(
        "Taken together, these analyses provide "
        + qualitative
        + " The Likert-scale response reflects both the size and "
        "consistency of the genus effects as well as their statistical "
        "significance."
    )

    explanation = " ".join(lines)

    conclusion = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

