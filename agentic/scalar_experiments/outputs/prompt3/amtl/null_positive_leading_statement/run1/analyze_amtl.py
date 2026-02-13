import json
from textwrap import dedent

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic preprocessing
    df = df.copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Identify the human genus label
    unique_genera = sorted(df["genus"].unique())
    human_candidates = [g for g in unique_genera if "Homo" in g]
    if not human_candidates:
        raise ValueError(f"No human genus found in data; genera present: {unique_genera}")
    human_genus = human_candidates[0]
    non_human_genera = [g for g in unique_genera if g != human_genus]

    # Fit binomial regression using proportions with socket counts as frequency weights
    formula = (
        f"prop_amtl ~ C(genus, Treatment(reference='{human_genus}'))"
        " + age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract genus effects (non-human vs human reference)
    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues

    genus_effects = {}
    for name in params.index:
        if name.startswith("C(genus"):
            if "[T." in name:
                genus_name = name.split("[T.", 1)[1].rstrip("]")
                genus_effects[genus_name] = {
                    "coef": float(params[name]),
                    "pvalue": float(pvalues[name]),
                    "ci_low": float(conf_int.loc[name, 0]),
                    "ci_high": float(conf_int.loc[name, 1]),
                }

    # Sanity check: ensure we have all non-human genera in the model
    for g in non_human_genera:
        if g not in genus_effects:
            raise ValueError(
                f"Did not find regression term for genus '{g}' in fitted model."
            )

    # Derive effect sizes (odds ratios) and characterize evidence
    for g, eff in genus_effects.items():
        beta = eff["coef"]
        eff["odds_ratio"] = float(np.exp(beta))
        eff["ci_low_or"] = float(np.exp(eff["ci_low"]))
        eff["ci_high_or"] = float(np.exp(eff["ci_high"]))

    # Determine overall direction of evidence:
    # coef < 0 => non-human genus has LOWER odds of AMTL than humans (supports "Yes")
    # coef > 0 => non-human genus has HIGHER odds of AMTL than humans (supports "No")
    evidence_scores = []
    for g in non_human_genera:
        eff = genus_effects[g]
        beta = eff["coef"]
        pval = eff["pvalue"]

        if beta == 0:
            direction = 0.0
        elif beta < 0:
            direction = 1.0  # supports "Yes"
        else:
            direction = -1.0  # supports "No"

        # Scale magnitude of effect: cap |beta| at 1.5 to avoid extremes
        magnitude = min(1.5, abs(beta)) / 1.5

        # Significance factor based on p-value
        if pval < 0.001:
            sig_factor = 1.0
        elif pval < 0.01:
            sig_factor = 0.8
        elif pval < 0.05:
            sig_factor = 0.6
        elif pval < 0.1:
            sig_factor = 0.4
        else:
            sig_factor = 0.2

        evidence_scores.append(direction * magnitude * sig_factor)

    # Aggregate evidence across genera
    if evidence_scores:
        overall_score = float(np.mean(evidence_scores))
    else:
        overall_score = 0.0

    if overall_score > 0:
        response = "Yes"
    elif overall_score < 0:
        response = "No"
    else:
        # Completely ambiguous signal
        response = "No"

    strength = int(round(min(100, max(0.0, abs(overall_score) * 100.0))))

    # Confidence heuristic based on consistency and significance
    betas = [genus_effects[g]["coef"] for g in non_human_genera]
    pvals = [genus_effects[g]["pvalue"] for g in non_human_genera]

    if response == "Yes":
        directional_flags = [b < 0 for b in betas]
    else:
        directional_flags = [b > 0 for b in betas]

    consistent_direction = all(directional_flags) if directional_flags else False
    min_p = min(pvals) if pvals else 1.0
    max_p = max(pvals) if pvals else 1.0

    if consistent_direction and min_p < 0.001:
        confidence = 90
    elif consistent_direction and min_p < 0.01:
        confidence = 80
    elif consistent_direction and min_p < 0.05:
        confidence = 70
    elif consistent_direction and min_p < 0.1:
        confidence = 60
    else:
        confidence = 50

    # Build explanation text
    n_rows = int(len(df))
    n_specimens = int(df["specimen"].nunique())

    expl_lines = []
    expl_lines.append(
        f"I analyzed {n_rows} tooth-class observations from {n_specimens} specimens "
        f"across the genera {', '.join(unique_genera)}, using modern humans "
        f"({human_genus}) as the reference group."
    )
    expl_lines.append(
        "I fit a binomial regression model for the proportion of antemortem tooth loss "
        "(number of missing teeth divided by observable sockets) with a logit link, "
        "including predictors for genus, age at death, estimated sex (probability of being male), "
        "and tooth class (anterior, posterior, premolar), and used socket counts as frequency weights."
    )

    for g in non_human_genera:
        eff = genus_effects[g]
        expl_lines.append(
            f"Compared with modern humans, the genus {g} had a log-odds difference in AMTL "
            f"of {eff['coef']:.3f} (odds ratio {eff['odds_ratio']:.2f}, "
            f"95% CI for the odds ratio [{eff['ci_low_or']:.2f}, {eff['ci_high_or']:.2f}], "
            f"p = {eff['pvalue']:.3g})."
        )

    if response == "Yes":
        direction_sentence = (
            "Across all non-human genera, the regression coefficients are predominantly "
            "negative, indicating lower odds of AMTL in non-human primates than in humans "
            "after adjusting for age, sex, and tooth class. This pattern supports the claim "
            "that modern humans have higher frequencies of AMTL than the non-human primate "
            "genera considered."
        )
    else:
        direction_sentence = (
            "The estimated genus effects do not consistently show lower odds of AMTL for "
            "non-human primates relative to humans, or the differences are small and/or "
            "statistically uncertain. Taken together, this does not provide strong support "
            "for humans having higher AMTL frequencies than all non-human genera once age, "
            "sex, and tooth class are accounted for."
        )

    expl_lines.append(direction_sentence)

    expl_lines.append(
        f"Based on these results, I answer the research question with '{response}', "
        f"with a strength rating of {strength} out of 100 and a confidence rating of "
        f"{confidence} out of 100, reflecting both the magnitude and statistical "
        "certainty of the genus effects."
    )

    explanation = " ".join(expl_lines)
    explanation = dedent(explanation).strip()

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

