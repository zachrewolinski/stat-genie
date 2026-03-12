import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    cwd = Path(".")
    info = load_metadata(cwd / "info.json")

    # Load dataset
    df = pd.read_csv(cwd / "amtl.csv")

    # Basic cleaning / checks
    df = df.copy()
    df = df[df["sockets"] > 0].reset_index(drop=True)

    # Proportion of antemortem tooth loss
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Ensure genus and tooth_class are treated as categorical
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Quick descriptive statistics: mean AMTL proportion by genus
    genus_prop = (
        df.groupby("genus")
        .apply(lambda x: np.average(x["prop_amtl"], weights=x["sockets"]))
        .to_dict()
    )

    # Binomial regression model with Homo sapiens as reference genus.
    # Response is proportion of missing teeth, with sockets as binomial trials.
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

    # Extract coefficients for each non-human genus relative to Homo sapiens
    coef = result.params
    pvals = result.pvalues

    genus_levels = [g for g in df["genus"].cat.categories if g != "Homo sapiens"]
    genus_effects = {}
    for g in genus_levels:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{g}]"
        if term in coef.index:
            genus_effects[g] = {
                "coef": float(coef[term]),
                "pvalue": float(pvals[term]),
            }

    # Compute adjusted predicted probabilities for each genus at typical covariate values
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    tooth_class_levels = df["tooth_class"].cat.categories.tolist()

    def predicted_prob_for_genus(genus_name: str) -> float:
        # Create a small design set for each tooth class, then average.
        rows = []
        for tc in tooth_class_levels:
            rows.append(
                {
                    "genus": genus_name,
                    "age": mean_age,
                    "prob_male": mean_prob_male,
                    "tooth_class": tc,
                }
            )
        new = pd.DataFrame(rows)
        preds = result.predict(new)
        return float(preds.mean())

    predicted_by_genus = {
        g: predicted_prob_for_genus(g) for g in df["genus"].cat.categories
    }

    # Determine overall evidence that humans have higher AMTL than non-human primates.
    human_genus = "Homo sapiens"
    human_pred = predicted_by_genus[human_genus]
    nonhuman_pred_values = [
        p for g, p in predicted_by_genus.items() if g != human_genus
    ]
    mean_nonhuman_pred = float(np.mean(nonhuman_pred_values))

    # Summarize significance across non-human genera
    significant_negative = []
    non_significant = []
    alpha = 0.05
    for g, eff in genus_effects.items():
        if eff["coef"] < 0 and eff["pvalue"] < alpha:
            significant_negative.append(g)
        else:
            non_significant.append(g)

    # Map evidence strength to a 0–100 Likert response.
    # Start from a neutral baseline and adjust based on:
    # - direction and significance of genus effects
    # - magnitude of predicted probability difference.
    diff = human_pred - mean_nonhuman_pred

    if significant_negative and not non_significant and diff > 0:
        # All non-human genera show significantly lower AMTL than humans
        # and humans have a higher adjusted predicted probability.
        if diff >= 0.10:
            likert = 90
        elif diff >= 0.05:
            likert = 80
        else:
            likert = 75
        answer = "Yes"
    elif diff > 0 and significant_negative:
        # Mixed significance but humans still clearly higher on average.
        if diff >= 0.10:
            likert = 75
        elif diff >= 0.05:
            likert = 70
        else:
            likert = 65
        answer = "Yes"
    elif diff > 0 and not significant_negative:
        # Humans somewhat higher but without strong statistical support.
        likert = 60
        answer = "Yes (weak evidence)"
    elif diff <= 0 and significant_negative:
        # Model says some non-human genera have higher or similar AMTL,
        # contradicting the prior belief.
        likert = 30
        answer = "No"
    else:
        # No clear evidence either way.
        likert = 50
        answer = "Inconclusive"

    # Build narrative explanation.
    explanation_lines = []
    explanation_lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher "
        "frequencies of antemortem tooth loss (AMTL) than non-human primate "
        "genera (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?"
    )
    explanation_lines.append(
        "I analyzed the dataset using a binomial regression (logistic GLM) where "
        "the response was the proportion of missing teeth (num_amtl / sockets) "
        "for each specimen and tooth class, with the number of sockets as the "
        "binomial trial count."
    )
    explanation_lines.append(
        "Predictors included genus (with Homo sapiens as the reference category), "
        "age at death, estimated probability of being male (prob_male), and "
        "tooth class (anterior, posterior, premolar)."
    )
    explanation_lines.append(
        "This model estimates, for each genus, the log-odds of AMTL while statistically "
        "controlling for age, sex, and tooth class."
    )
    explanation_lines.append(
        f"Crude (weighted) mean AMTL proportions by genus were: {genus_prop}."
    )
    explanation_lines.append(
        "From the regression, each non-human genus has a coefficient that represents "
        "its difference in log-odds of AMTL relative to Homo sapiens; negative values "
        "indicate lower AMTL than humans."
    )
    explanation_lines.append(
        f"Estimated genus effects (non-human vs human) were: {genus_effects}."
    )
    explanation_lines.append(
        "To interpret the model on the probability scale, I computed adjusted predicted "
        "AMTL probabilities for each genus at the mean age, mean sex estimate, and "
        "averaged across tooth classes."
    )
    explanation_lines.append(
        f"These adjusted predicted AMTL probabilities by genus were: {predicted_by_genus}."
    )
    explanation_lines.append(
        f"Humans had an adjusted AMTL probability of approximately {human_pred:.3f}, "
        f"while the mean across non-human genera was about {mean_nonhuman_pred:.3f}, "
        f"giving a difference of {diff:.3f} in absolute probability."
    )
    explanation_lines.append(
        f"Across non-human genera, those with significantly negative coefficients "
        f"(lower AMTL than humans at p < 0.05) were: {significant_negative}, "
        f"while non-significant or positive differences were: {non_significant}."
    )
    if answer.startswith("Yes"):
        explanation_lines.append(
            "Taken together, the direction of the genus coefficients, their statistical "
            "significance, and the adjusted predicted probabilities indicate that "
            "modern humans do have higher frequencies of AMTL than the non-human "
            "primate genera in this dataset after accounting for age, sex, and tooth class."
        )
    elif answer == "No":
        explanation_lines.append(
            "Taken together, the genus coefficients and adjusted predicted probabilities "
            "do not support the claim that humans have higher AMTL; instead, some "
            "non-human genera show equal or higher AMTL when age, sex, and tooth "
            "class are controlled for."
        )
    else:
        explanation_lines.append(
            "Taken together, the genus coefficients and adjusted predicted probabilities "
            "do not provide clear, statistically robust evidence that humans have "
            "higher AMTL than non-human primates once age, sex, and tooth class "
            "are controlled for."
        )
    explanation_lines.append(
        f"On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"I summarize the strength of evidence for the claim as {likert}, corresponding "
        f"to an overall conclusion of '{answer}'."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(likert), "explanation": explanation}

    # Write the required JSON object to conclusion.txt
    conclusion_path = cwd / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

