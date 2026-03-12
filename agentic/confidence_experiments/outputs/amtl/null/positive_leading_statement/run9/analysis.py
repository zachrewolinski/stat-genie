import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Total potential teeth in a class = observed sockets + missing teeth
    df["total_teeth"] = df["num_amtl"] + df["sockets"]
    # Exclude any rows with non-positive totals just in case
    df = df[df["total_teeth"] > 0].copy()

    # Proportion of teeth lost
    df["prop_amtl"] = df["num_amtl"] / df["total_teeth"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = df["genus"].isin(["Homo sapiens", "Homo"]).astype(int)

    # Design matrix: human vs non-human + age, sex proxy, and tooth class
    covariates = df[["is_human", "age", "prob_male", "tooth_class"]].copy()
    covariates = pd.get_dummies(covariates, columns=["tooth_class"], drop_first=True)
    X = sm.add_constant(covariates)

    # Binomial regression with aggregated counts: use proportion + total as frequency weights
    model = sm.GLM(
        df["prop_amtl"],
        X,
        family=sm.families.Binomial(),
        freq_weights=df["total_teeth"],
    )
    result = model.fit()

    # Extract human effect
    human_coef = float(result.params["is_human"])
    human_pval = float(result.pvalues["is_human"])
    human_or = float(np.exp(human_coef))

    # Predicted probabilities for a "typical" case:
    # use mean age, mean prob_male, and the most common tooth_class
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    most_common_class = df["tooth_class"].mode().iat[0]

    def build_row(is_human: int) -> pd.DataFrame:
        row = {
            "const": 1.0,
            "is_human": is_human,
            "age": mean_age,
            "prob_male": mean_prob_male,
        }
        # Handle tooth_class dummies consistently with the fitted model
        for col in X.columns:
            if col.startswith("tooth_class_"):
                row[col] = 1.0 if col == f"tooth_class_{most_common_class}" else 0.0
        return pd.DataFrame([row], columns=X.columns)

    pred_nonhuman = float(result.predict(build_row(is_human=0))[0])
    pred_human = float(result.predict(build_row(is_human=1))[0])
    diff_pred = pred_human - pred_nonhuman

    # Map statistical evidence to a 0–100 Likert-style response
    # Heuristic:
    # - If p >= 0.05, treat as "No" with low confidence.
    # - If p < 0.05 and human_or > 1, treat as "Yes" with strength scaled
    #   by both effect size and significance.
    if human_pval >= 0.05:
        response = 20
        verdict = "No"
    else:
        # Base strength from odds ratio (capped for stability)
        effect_strength = min(max(human_or - 1.0, 0.0), 3.0) / 3.0
        # Stronger evidence for smaller p-values
        sig_strength = min(-np.log10(max(human_pval, 1e-16)) / 10.0, 1.0)
        strength = max(min(0.4 + 0.6 * (0.5 * effect_strength + 0.5 * sig_strength), 1.0), 0.0)
        response = int(round(50 + 50 * strength))
        verdict = "Yes" if human_or > 1.0 else "No"

    # Compose explanation text
    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio) "
        "after accounting for age, sex, and tooth class?\n\n"
        "I modeled AMTL using a binomial regression on the proportion of missing teeth "
        "within each specimen and tooth class: num_amtl / (num_amtl + sockets). "
        "The model included a binary indicator for modern humans vs non-human primates, "
        "age at death, estimated probability of being male, and tooth-class indicators "
        "as predictors. The binomial model was fitted using statsmodels GLM with the "
        "total number of potential teeth in each row as the binomial trial count.\n\n"
        f"The coefficient for modern humans (vs non-human primates) was {human_coef:.3f}, "
        f"corresponding to an odds ratio of {human_or:.2f} for AMTL, with a p-value of "
        f"{human_pval:.3g}. For a typical specimen (mean age, mean sex estimate, and the "
        f"most common tooth class), the predicted AMTL frequency was "
        f"{pred_nonhuman:.3%} for non-human primates and {pred_human:.3%} for modern "
        f"humans, a difference of {diff_pred:.3%}.\n\n"
        f"Given this model, the data {'do' if verdict == 'Yes' else 'do not'} provide "
        f"statistically significant evidence that modern humans have higher AMTL "
        f"frequencies than non-human primates after accounting for age, sex, and tooth "
        f"class. The assigned Likert-scale response value of {response} reflects the "
        f"overall strength of this evidence and effect size in support of a '{verdict}' "
        "answer to the research question."
    )

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

