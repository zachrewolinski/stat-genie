import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Proportion of missing teeth for each specimen/tooth class row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Identify the human genus label programmatically
    genera = sorted(df["genus"].unique())
    human_candidates = [g for g in genera if "Homo" in g]
    if not human_candidates:
        raise ValueError("Could not identify human genus in data.")
    human_genus = human_candidates[0]
    nonhuman_genera = [g for g in genera if g != human_genus]

    # Binomial GLM: proportion missing ~ genus + age + sex + tooth_class
    formula = (
        f'prop_amtl ~ C(genus, Treatment(reference="{human_genus}")) '
        "+ age + prob_male + C(tooth_class)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Raw (unadjusted) AMTL frequencies by genus
    observed_rates = {}
    for g in genera:
        mask = df["genus"] == g
        num_missing = df.loc[mask, "num_amtl"].sum()
        num_sockets = df.loc[mask, "sockets"].sum()
        observed_rates[g] = float(num_missing) / float(num_sockets)

    # Model-adjusted AMTL frequencies by genus
    adjusted_rates = {}
    for g in genera:
        df_g = df.copy()
        df_g["genus"] = g
        preds = result.predict(df_g)
        adjusted_rates[g] = float(np.average(preds, weights=df_g["sockets"]))

    human_adj = adjusted_rates[human_genus]
    nonhuman_adj_values = [adjusted_rates[g] for g in nonhuman_genera]

    # Decide whether humans have higher AMTL than all non-human genera
    humans_higher = human_adj > max(nonhuman_adj_values)
    response = "Yes" if humans_higher else "No"

    # Simple confidence heuristic based on genus coefficients and direction
    genus_params = {
        name: coef
        for name, coef in result.params.items()
        if name.startswith("C(genus")
    }
    genus_pvalues = {
        name: pval
        for name, pval in result.pvalues.items()
        if name.startswith("C(genus")
    }

    # For humans_higher == False, positive genus coefficients mean
    # higher AMTL in non-human genera relative to humans.
    sig_nonhuman_gt_human = sum(
        1
        for name, coef in genus_params.items()
        if coef > 0 and genus_pvalues.get(name, 1.0) < 0.05
    )
    sig_nonhuman_diff_any = sum(
        1
        for name, coef in genus_params.items()
        if genus_pvalues.get(name, 1.0) < 0.05
    )

    if humans_higher:
        # If humans are estimated higher but contrasts are weak, lower confidence.
        confidence = 75
        if sig_nonhuman_diff_any == 0:
            confidence = 65
        elif sig_nonhuman_diff_any >= len(nonhuman_genera):
            confidence = 85
    else:
        # Humans not higher than at least one non-human genus.
        confidence = 80
        if sig_nonhuman_gt_human == 0:
            confidence = 70
        elif sig_nonhuman_gt_human >= 1:
            confidence = 90

    def format_rates(rates: dict) -> str:
        parts = [f"{g}: {rate * 100:.1f}%" for g, rate in rates.items()]
        return ", ".join(parts)

    observed_str = format_rates(observed_rates)
    adjusted_str = format_rates(adjusted_rates)

    if humans_higher:
        comparison_clause = (
            f"The adjusted AMTL frequency for modern humans ({human_genus}) "
            f"({human_adj * 100:.1f}%) is higher than for all non-human genera "
            f"({', '.join(f'{g} {adjusted_rates[g] * 100:.1f}%' for g in nonhuman_genera)}). "
        )
    else:
        comparison_clause = (
            f"The adjusted AMTL frequency for modern humans ({human_genus}) "
            f"({human_adj * 100:.1f}%) is not higher than that of the non-human genera "
            f"({', '.join(f'{g} {adjusted_rates[g] * 100:.1f}%' for g in nonhuman_genera)}). "
        )

    if sig_nonhuman_gt_human > 0:
        significance_clause = (
            f"In the regression, {sig_nonhuman_gt_human} non-human genus"
            f"{'es' if sig_nonhuman_gt_human > 1 else ''} show significantly higher AMTL "
            "than humans at the 0.05 level, reinforcing this conclusion."
        )
    elif sig_nonhuman_diff_any > 0:
        significance_clause = (
            "Some genus contrasts are statistically significant at the 0.05 level, "
            "indicating real differences in AMTL frequencies across genera."
        )
    else:
        significance_clause = (
            "Genus coefficients are not strongly significant at the 0.05 level, "
            "so this conclusion is based on effect sizes rather than formal significance."
        )

    explanation = (
        "I modeled antemortem tooth loss (AMTL) as the proportion of missing teeth "
        "(num_amtl / sockets) for each specimen and tooth class using a binomial "
        "logistic regression with genus, age at death, estimated sex (prob_male), "
        "and tooth class as predictors, and the number of observable sockets as the "
        "binomial denominator. "
        f"Raw AMTL frequencies (missing teeth / observable sockets) by genus are: {observed_str}. "
        f"Model-adjusted AMTL frequencies, averaging over the observed distribution of age, sex, "
        f"and tooth class, are: {adjusted_str}. "
        + comparison_clause
        + significance_clause
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

