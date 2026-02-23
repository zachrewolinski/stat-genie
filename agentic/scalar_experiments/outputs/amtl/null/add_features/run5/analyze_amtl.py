import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


def map_evidence_to_likert(beta: float, pval: float) -> int:
    """
    Map the evidence about the human effect (beta, p-value)
    to a 0–100 Likert scale where:
    - 0   = strong "No" (strong evidence against humans having higher AMTL)
    - 50  = ambiguous/very weak evidence either way
    - 100 = strong "Yes" (strong evidence that humans have higher AMTL)
    Values < 50 correspond to a "No" answer to the question
    "Do humans have higher AMTL?", and values > 50 correspond to "Yes".
    """
    # Determine directional "yes" (higher AMTL for humans) if beta > 0 and significant.
    is_yes_direction = beta > 0 and pval < 0.05

    # Categorize strength of evidence mainly from the p-value.
    if pval >= 0.1 or np.isnan(pval):
        base = 50  # essentially no evidence either way
    elif pval >= 0.05:
        base = 55  # very weak evidence
    elif pval >= 0.01:
        base = 70  # moderate evidence
    elif pval >= 0.001:
        base = 80  # strong evidence
    else:
        base = 90  # very strong evidence

    # If direction supports "Yes", keep base as-is (>50). Otherwise mirror below 50.
    if is_yes_direction:
        response_value = base
    else:
        response_value = 100 - base

    return int(max(0, min(100, round(response_value))))


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Basic cleaning and restriction to the four genera of interest
    required_cols = [
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "genus",
        "tooth_class",
        "specimen",
    ]
    df = df.dropna(subset=required_cols)
    df = df[df["sockets"] > 0]
    df = df[(df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]

    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Construct key variables
    df["is_human"] = (df["genus"].str.strip() == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    n_total = int(df.shape[0])
    n_human = int(df["is_human"].sum())
    n_nonhuman = int(n_total - n_human)

    mean_prop_by_genus: Dict[str, float] = (
        df.groupby("genus")["prop_amtl"].mean().to_dict()
    )

    # Design matrix for binomial logistic regression
    X = pd.DataFrame(
        {
            "intercept": 1.0,
            "is_human": df["is_human"].astype(float),
            "age": df["age"].astype(float),
            "prob_male": df["prob_male"].astype(float),
        }
    )
    tooth_dummies = pd.get_dummies(
        df["tooth_class"], prefix="tooth", drop_first=True
    )
    X = pd.concat([X, tooth_dummies], axis=1)

    # Binomial response: successes = num_amtl, failures = sockets - num_amtl
    y = np.column_stack(
        [df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()]
    )

    # Fit GLM with binomial family
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    # Extract coefficient and uncertainty for the human effect
    beta = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])

    ci_low, ci_high = result.conf_int().loc["is_human"].tolist()
    or_human = float(np.exp(beta))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    # Predicted probabilities for humans vs non-humans with covariates held at observed values
    X_human = X.copy()
    X_human["is_human"] = 1.0
    X_non = X.copy()
    X_non["is_human"] = 0.0

    pred_human = result.predict(X_human)
    pred_non = result.predict(X_non)

    mean_pred_human = float(pred_human.mean())
    mean_pred_non = float(pred_non.mean())
    diff_pred = mean_pred_human - mean_pred_non

    # Decide Yes/No based on sign and significance of human effect
    if pval < 0.05 and beta > 0:
        yes_no_answer = "Yes"
        qualitative = "do"
    else:
        yes_no_answer = "No"
        qualitative = "do not"

    response_value = map_evidence_to_likert(beta=beta, pval=pval)

    # Build human-readable explanation
    genus_lines = []
    for genus, mean_prop in mean_prop_by_genus.items():
        genus_lines.append(
            f"- {genus}: mean AMTL proportion {mean_prop:.3f}"
        )
    genus_summary = "\n".join(genus_lines)

    explanation = f"""
    Research question: Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) after accounting for age, sex, and tooth class?

    Data and design:
    - Analyzed {n_total} genus–tooth-class observations with non-missing values on AMTL, sockets, age, sex, and tooth class.
    - Sample sizes: {n_human} human observations and {n_nonhuman} non-human observations drawn from the four genera of interest.
    - Outcome: number of missing teeth of a given class (num_amtl) out of the number of observable sockets (sockets), modeled as a binomial proportion.
    - Predictors: human vs non-human indicator (is_human), age at death (age), probability of being male (prob_male), and tooth class (anterior/posterior/premolar).

    Descriptive patterns (raw, unadjusted):
    {genus_summary}

    Modeling approach:
    - Fit a binomial logistic regression for the AMTL proportion with a logit link, using num_amtl successes out of sockets trials.
    - Included human vs non-human status, age, sex (prob_male), and tooth-class indicators as predictors.
    - Inference is based on the model's standard (maximum-likelihood) standard errors.

    Key results for the human vs non-human contrast:
    - Log-odds coefficient for humans (vs non-humans): {beta:.3f}.
    - Odds ratio for AMTL in humans relative to non-humans: {or_human:.3f} (95% CI: {or_low:.3f}–{or_high:.3f}).
    - p-value for the human coefficient: {pval:.4g}.
    - Model-based mean predicted AMTL proportion (holding age, sex, and tooth class at their observed values but toggling human status):
      * Humans: {mean_pred_human:.3f}
      * Non-humans: {mean_pred_non:.3f}
      * Difference (human minus non-human): {diff_pred:.3f}.

    Interpretation:
    - Based on the sign and significance of the human coefficient, the answer to the research question is: {yes_no_answer} — humans {qualitative} show statistically higher AMTL frequencies than the non-human primate genera once age, sex, and tooth class are accounted for.
    - The Likert-scale response of {response_value} (0 = strong "No", 100 = strong "Yes") reflects both the direction of the estimated human effect and the strength of the statistical evidence summarized above.
    """

    explanation_clean = "\n".join(
        line.rstrip() for line in explanation.strip().splitlines()
    )

    output = {
        "response": response_value,
        "explanation": explanation_clean,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
