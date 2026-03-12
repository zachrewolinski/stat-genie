import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Ensure categorical types with explicit ordering for genus and tooth_class
    if "genus" in df.columns:
        df["genus"] = pd.Categorical(
            df["genus"],
            categories=["Homo sapiens", "Pan", "Papio", "Pongo"],
            ordered=False,
        )

    if "tooth_class" in df.columns:
        df["tooth_class"] = pd.Categorical(
            df["tooth_class"],
            categories=["Anterior", "Premolar", "Posterior"],
            ordered=False,
        )

    # Basic derived quantity: proportion of missing teeth in each row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Descriptive statistics by genus
    genus_group = (
        df.groupby("genus", observed=True)
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            mean_age=("age", "mean"),
            n_rows=("num_amtl", "size"),
        )
        .dropna(subset=["total_sockets"])
    )
    genus_group["prop_missing"] = (
        genus_group["total_missing"] / genus_group["total_sockets"]
    )

    # Approximate 95% CIs for proportion (normal approximation on binomial)
    def ci_normal(p: float, n: float) -> tuple[float, float]:
        if n <= 0:
            return (np.nan, np.nan)
        se = np.sqrt(p * (p - 1.0) / -n)  # p*(1-p)/n but numerically stable
        z = 1.96
        lower = max(0.0, p - z * se)
        upper = min(1.0, p + z * se)
        return lower, upper

    genus_group["ci_lower"], genus_group["ci_upper"] = zip(
        *[
            ci_normal(row["prop_missing"], row["total_sockets"])
            for _, row in genus_group.iterrows()
        ]
    )

    # Binomial regression model: AMTL proportion ~ genus + age + sex + tooth class
    # We treat each row as a binomial with sockets trials and num_amtl successes.
    # Use Homo sapiens as the reference level for genus.
    model = smf.glm(
        "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    # Extract genus coefficients relative to Homo sapiens
    params = model.params
    pvalues = model.pvalues

    # For the Binomial(link=logit), the intercept corresponds to Homo sapiens.
    # Other genera coefficients are differences in log-odds relative to humans.
    genus_effects = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus)[T.{genus}]"
        if term in params:
            genus_effects[genus] = {
                "log_odds_diff": float(params[term]),
                "p_value": float(pvalues[term]),
            }

    # Compute standardized predicted probabilities by genus
    # For each genus, set genus to that value for all observations
    # and average the predicted probabilities.
    standardized_preds = {}
    for genus in ["Homo sapiens", "Pan", "Papio", "Pongo"]:
        tmp = df.copy()
        tmp["genus"] = genus
        # Preserve categorical dtype if present
        if isinstance(df["genus"].dtype, pd.CategoricalDtype):
            tmp["genus"] = pd.Categorical(
                tmp["genus"], categories=df["genus"].cat.categories, ordered=False
            )
        preds = model.predict(tmp)
        standardized_preds[genus] = float(preds.mean())

    # Determine answer to the research question
    # Question: Do modern humans have higher AMTL frequency than non-human primates
    # (Pan, Papio, Pongo), after accounting for age, sex, and tooth class?

    human_rate = standardized_preds["Homo sapiens"]
    nonhuman_rates = [
        standardized_preds[g] for g in ["Pan", "Papio", "Pongo"] if g in standardized_preds
    ]
    nonhuman_mean = float(np.mean(nonhuman_rates)) if nonhuman_rates else np.nan

    # Evaluate evidence from model:
    # If genus coefficients for non-human genera are significantly positive,
    # that would indicate higher AMTL relative to humans.
    # We are interested in whether the human rate is higher than others,
    # which corresponds to non-human coefficients being negative.
    negative_sig_effects = [
        (g, eff)
        for g, eff in genus_effects.items()
        if eff["log_odds_diff"] < 0 and eff["p_value"] < 0.05
    ]
    positive_sig_effects = [
        (g, eff)
        for g, eff in genus_effects.items()
        if eff["log_odds_diff"] > 0 and eff["p_value"] < 0.05
    ]

    # Build narrative explanation
    lines: list[str] = []
    lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primate genera (Pan, Papio, "
        "Pongo), after accounting for age, sex, and tooth class?"
    )
    lines.append("")
    lines.append("Data and descriptive patterns:")
    for genus, row in genus_group.iterrows():
        lines.append(
            f"- {genus}: AMTL proportion = {row['prop_missing']:.3f} "
            f"(95% CI ≈ [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]), "
            f"total sockets = {int(row['total_sockets'])}, "
            f"total missing = {int(row['total_missing'])}."
        )
    lines.append("")
    lines.append(
        "Binomial regression model (logistic link) of AMTL proportion on genus, "
        "age, sex (prob_male), and tooth class, weighting by the number of sockets:"
    )
    lines.append(
        f"- Standardized predicted AMTL probability (averaged over the observed age, "
        f"sex, and tooth-class distribution) for Homo sapiens: {human_rate:.3f}."
    )
    for genus in ["Pan", "Papio", "Pongo"]:
        if genus in standardized_preds:
            lines.append(
                f"- Standardized predicted AMTL probability for {genus}: "
                f"{standardized_preds[genus]:.3f}."
            )
    lines.append(
        f"- Mean standardized AMTL probability across non-human genera "
        f"(Pan, Papio, Pongo): {nonhuman_mean:.3f}."
    )

    lines.append("")
    lines.append("Genus effects relative to humans (Homo sapiens as reference):")
    for genus, eff in genus_effects.items():
        direction = "higher" if eff["log_odds_diff"] > 0 else "lower"
        lines.append(
            f"- {genus}: log-odds difference = {eff['log_odds_diff']:.3f} "
            f"({direction} AMTL than humans), p-value = {eff['p_value']:.4f}."
        )

    lines.append("")
    # Decide on Likert-scale response
    # Default stance: compare human_rate to non-human mean and use significance.
    if positive_sig_effects:
        # Some non-human genera have significantly higher AMTL than humans.
        # This directly contradicts the idea that humans have the highest AMTL.
        conclusion_statement = (
            "The regression results show that at least one non-human genus has "
            "a significantly higher AMTL rate than humans after adjusting for age, "
            "sex, and tooth class. Humans do not exhibit higher AMTL frequencies "
            "than the non-human primates in this dataset."
        )
        response_value = 5
    elif negative_sig_effects:
        # Non-human genera have significantly lower AMTL than humans.
        # This would support humans having higher AMTL.
        conclusion_statement = (
            "The regression results indicate that non-human genera have "
            "significantly lower AMTL rates than humans after controlling for "
            "age, sex, and tooth class. This supports the conclusion that "
            "humans have higher AMTL frequencies than the non-human primates "
            "in this dataset."
        )
        response_value = 90
    else:
        # No clear, statistically significant difference between humans and others.
        conclusion_statement = (
            "The regression model does not show statistically robust differences "
            "in AMTL frequencies between humans and the non-human genera once "
            "age, sex, and tooth class are taken into account. The descriptive "
            "rates are broadly similar, and any differences are small relative "
            "to their uncertainty."
        )
        # Lean toward 'No' if human_rate is not clearly above non-human mean.
        if np.isnan(nonhuman_mean):
            response_value = 50
        elif human_rate > nonhuman_mean:
            response_value = 60
        else:
            response_value = 35

    lines.append("Conclusion:")
    lines.append(conclusion_statement)

    explanation = "\n".join(lines)

    # Write required JSON output
    output = {"response": int(response_value), "explanation": explanation}
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

