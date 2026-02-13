import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df = df.copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Ensure Homo sapiens is the reference genus
    genus_order = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    present_genera = [g for g in genus_order if g in set(df["genus"])]
    df["genus"] = df["genus"].cat.reorder_categories(present_genera, ordered=False)

    # Descriptive AMTL proportions by genus
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(prop=lambda d: d["num_amtl"] / d["sockets"])
    )

    # Binomial regression: proportion with binomial family and socket counts as weights
    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    # Extract effects for non-human genera relative to Homo sapiens
    genus_effects = []
    for name, coef in params.items():
        if "C(genus" in name and "[T." in name:
            pval = float(pvalues[name])
            ci_low, ci_high = map(float, conf_int.loc[name])
            genus_name = name.split("[T.")[-1].rstrip("]")
            genus_effects.append(
                {
                    "genus": genus_name,
                    "coef": float(coef),
                    "pval": pval,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    # Determine whether humans have higher AMTL frequency than all non-human genera
    # after adjusting for covariates. With Homo sapiens as the reference, a negative
    # and statistically significant coefficient for each non-human genus implies
    # lower AMTL odds in that genus relative to humans.
    higher_than_all = True
    significantly_higher_than_all = True
    for eff in genus_effects:
        coef = eff["coef"]
        pval = eff["pval"]
        # Non-human genus has lower AMTL if its coefficient is negative
        higher_than_this = coef < 0.0
        significant_diff = (coef < 0.0) and (pval < 0.05)
        higher_than_all &= higher_than_this
        significantly_higher_than_all &= significant_diff

    if higher_than_all and significantly_higher_than_all:
        response = "Yes"
    else:
        response = "No"

    # Build a concise, data-driven explanation
    human_prop = float(genus_summary.loc["Homo sapiens", "prop"])
    nonhuman_lines = []
    for genus, row in genus_summary.iterrows():
        if genus == "Homo sapiens":
            continue
        nonhuman_lines.append(
            f"{genus}: AMTL proportion ≈ {row['prop']:.3f} "
            f"({int(row['num_amtl'])} missing teeth out of {int(row['sockets'])})"
        )
    nonhuman_desc = "; ".join(nonhuman_lines)

    effect_lines = []
    for eff in genus_effects:
        direction = "lower" if eff["coef"] < 0 else "higher"
        sig_word = "statistically significant" if eff["pval"] < 0.05 else "not statistically significant"
        effect_lines.append(
            f"{eff['genus']} has {direction} AMTL odds than humans "
            f"(log-odds difference {eff['coef']:.3f}, 95% CI [{eff['ci_low']:.3f}, {eff['ci_high']:.3f}], "
            f"p = {eff['pval']:.3g}; {sig_word})."
        )
    effects_desc = " ".join(effect_lines)

    explanation_parts = [
        "I modeled antemortem tooth loss (AMTL) using a binomial regression on the proportion of missing teeth (num_amtl / sockets) for each specimen,",
        "with genus, age at death, estimated sex (prob_male), and tooth class (anterior, premolar, posterior) as predictors.",
        f"Descriptively, humans show an overall AMTL proportion of approximately {human_prop:.3f}, while non-human genera show: {nonhuman_desc}.",
        f"In the regression with Homo sapiens as the reference category, the coefficients for the non-human genera quantify how their AMTL odds differ from those of humans after adjusting for age, sex, and tooth class: {effects_desc}",
    ]

    if response == "Yes":
        explanation_parts.append(
            "Because all non-human genera have negative and statistically significant coefficients, "
            "their AMTL odds are lower than those of humans after controlling for age, sex, and tooth class. "
            "This supports the conclusion that modern humans have higher frequencies of AMTL than the non-human primate genera in this dataset."
        )
    else:
        explanation_parts.append(
            "At least one non-human genus does not show a clear and statistically significant decrease in AMTL odds relative to humans "
            "after adjusting for age, sex, and tooth class, so the data do not provide strong evidence that humans have uniformly higher AMTL frequencies "
            "than all non-human primate genera in this sample."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

