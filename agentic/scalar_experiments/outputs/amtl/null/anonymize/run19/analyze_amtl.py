import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Rename columns to meaningful names
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing_teeth",
            "feature4": "observable_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Keep rows with valid counts
    df = df[df["observable_sockets"] > 0].copy()

    # Define response as proportion of missing teeth with number of trials as weights
    df["prop_missing"] = df["missing_teeth"] / df["observable_sockets"]

    # Drop any rows with missing values in key variables
    df = df.dropna(
        subset=[
            "prop_missing",
            "observable_sockets",
            "age",
            "sex_estimate",
            "tooth_class",
            "genus",
            "specimen_id",
        ]
    ).copy()

    # Ensure categorical variables are treated as such
    df["tooth_class"] = df["tooth_class"].astype("category")
    df["genus"] = df["genus"].astype("category")

    # Make Homo sapiens the reference category for genus so that
    # coefficients for other genera are relative to humans.
    if "Homo sapiens" in list(df["genus"].cat.categories):
        df["genus"] = df["genus"].cat.reorder_categories(
            sorted(df["genus"].cat.categories, key=lambda x: (x != "Homo sapiens", x)),
            ordered=True,
        )

    # Fit binomial GLM with logit link; use observable_sockets as frequency weights
    formula = "prop_missing ~ C(genus) + age + sex_estimate + C(tooth_class)"
    glm_model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable_sockets"],
    )

    # Cluster-robust standard errors by specimen to account for repeated measures
    glm_results = glm_model.fit(
        cov_type="cluster", cov_kwds={"groups": df["specimen_id"]}
    )

    # Compute marginal predicted AMTL frequency for each genus by averaging over
    # the empirical distribution of age, sex, and tooth class.
    genera = list(df["genus"].cat.categories)
    total_sockets = df["observable_sockets"].sum()

    genus_preds = {}
    for g in genera:
        df_g = df.copy()
        df_g["genus"] = g
        probs = glm_results.predict(df_g)
        # Socket-weighted average probability of AMTL
        weighted_prob = np.average(probs, weights=df["observable_sockets"])
        genus_preds[g] = float(weighted_prob)

    # Extract coefficients for non-human genera relative to Homo sapiens
    params = glm_results.params
    pvalues = glm_results.pvalues

    human_genus = "Homo sapiens"
    nonhuman_genera = [g for g in genera if g != human_genus]

    genus_effects = {}
    for g in nonhuman_genera:
        term = f"C(genus)[T.{g}]"
        if term in params.index:
            genus_effects[g] = {
                "coef_log_odds_diff_vs_human": float(params[term]),
                "p_value": float(pvalues[term]),
            }

    # Summarize whether humans have higher AMTL frequencies
    human_pred = genus_preds.get(human_genus, np.nan)
    diffs = {g: human_pred - genus_preds[g] for g in nonhuman_genera}

    # Heuristic strength assessment for Likert mapping:
    # - Base strength on (a) direction and magnitude of differences
    #   and (b) statistical significance of genus coefficients.
    significant_lower = [
        g
        for g in nonhuman_genera
        if genus_effects.get(g, {}).get("coef_log_odds_diff_vs_human", 0.0) < 0
        and genus_effects.get(g, {}).get("p_value", 1.0) < 0.05
    ]

    # Start from neutral (50) and adjust
    response_score: int
    if not np.isfinite(human_pred):
        response_score = 50
    else:
        avg_abs_diff = float(np.mean([abs(d) for d in diffs.values()]))

        # Map average absolute difference in predicted proportion to a rough scale
        # and upweight when most non-human genera are significantly lower than humans.
        base = 50.0
        if avg_abs_diff >= 0.10:
            base = 85.0
        elif avg_abs_diff >= 0.05:
            base = 75.0
        elif avg_abs_diff >= 0.02:
            base = 65.0
        elif avg_abs_diff >= 0.01:
            base = 60.0

        if len(significant_lower) >= 2:
            base += 10.0
        elif len(significant_lower) == 1:
            base += 5.0

        response_score = int(max(0, min(100, round(base))))

    # Build explanation string with key numerical results
    lines = []
    lines.append(
        "I fit a binomial logistic regression model for the proportion of missing teeth "
        "(missing_teeth / observable_sockets) with a logit link, using observable_sockets "
        "as frequency weights to reflect the number of teeth at risk."
    )
    lines.append(
        "The predictors were genus (Homo sapiens vs. Pan, Papio, Pongo), age at death, "
        "estimated sex, and tooth class (anterior, posterior, premolar). Cluster-robust "
        "standard errors were used with specimen ID as the clustering variable to account "
        "for multiple observations per individual."
    )

    # Add predicted frequencies by genus
    genus_summaries = []
    for g in genera:
        genus_summaries.append(f"{g}: {genus_preds[g]:.3f}")
    lines.append(
        "Socket-weighted average predicted AMTL frequencies (proportion of teeth missing) "
        f"by genus were: {', '.join(genus_summaries)}."
    )

    # Add coefficient and p-value summaries
    coef_summaries = []
    for g in nonhuman_genera:
        eff = genus_effects.get(g)
        if eff is not None:
            coef_summaries.append(
                f"{g} vs. Homo sapiens: log-odds difference = {eff['coef_log_odds_diff_vs_human']:.3f}, "
                f"p-value = {eff['p_value']:.4f}"
            )
    if coef_summaries:
        lines.append(
            "In the regression model with Homo sapiens as the reference genus, the coefficients "
            "for the non-human genera were: "
            + "; ".join(coef_summaries)
            + ". Negative coefficients indicate lower AMTL odds than humans."
        )

    if len(significant_lower) == len(nonhuman_genera) and len(nonhuman_genera) > 0:
        lines.append(
            "All non-human genera show significantly lower odds of AMTL than modern humans "
            "after adjusting for age, sex, and tooth class (p < 0.05 for each genus)."
        )
    elif len(significant_lower) > 0:
        lines.append(
            f"{len(significant_lower)} of {len(nonhuman_genera)} non-human genera have significantly lower "
            "AMTL odds than humans after adjustment (p < 0.05)."
        )
    else:
        lines.append(
            "None of the non-human genera show statistically significant AMTL differences relative to humans "
            "after adjustment at the 0.05 level."
        )

    if response_score >= 70:
        lines.append(
            f"Taken together, these results provide strong evidence that modern humans have higher AMTL "
            f"frequencies than the non-human primate genera considered here, leading to a 'Yes' answer with "
            f"strength {response_score} on a 0–100 Likert scale."
        )
    elif response_score <= 40:
        lines.append(
            f"Overall, the results do not support higher AMTL frequencies in humans relative to the non-human "
            f"primates, leading to a 'No' answer with strength {response_score} on a 0–100 Likert scale."
        )
    else:
        lines.append(
            f"The evidence for higher AMTL frequencies in humans relative to the non-human primates is mixed "
            f"or modest, leading to an equivocal answer with strength {response_score} on a 0–100 Likert scale."
        )

    conclusion = {
        "response": response_score,
        "explanation": " ".join(lines),
    }

    # Write conclusion.json-like object to conclusion.txt as required
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

