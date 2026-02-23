import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Prepare variables
    df = df.copy()
    df["missing"] = df["feature3"].astype(float)
    df["total"] = df["feature4"].astype(float)
    df = df[df["total"] > 0].copy()

    # Human vs non-human primate indicator
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Tooth class as categorical (Anterior, Posterior, Premolar)
    df["tooth_class"] = df["feature1"].astype("category")

    # Age and sex covariates
    df["age"] = df["feature5"].astype(float)
    df["sex_est"] = df["feature7"].astype(float)

    # Binomial response: [successes, failures]
    endog = np.column_stack([df["missing"], df["total"] - df["missing"]])

    # Design matrix: is_human + age + sex_est + tooth_class dummies
    exog_vars = ["is_human", "age", "sex_est", "tooth_class"]
    exog = pd.get_dummies(df[exog_vars], drop_first=True)
    exog = sm.add_constant(exog, has_constant="add")

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    # Extract human effect
    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    p_human = float(result.pvalues["is_human"])
    odds_ratio_human = float(np.exp(coef_human))

    # Predicted probabilities at typical covariate values
    median_age = float(df["age"].median())
    median_sex = float(df["sex_est"].median())

    # Build a reference row using the structure of exog
    base_vals = {col: 0.0 for col in exog.columns}
    base_vals["const"] = 1.0
    base_vals["is_human"] = 0.0
    if "age" in base_vals:
        base_vals["age"] = median_age
    if "sex_est" in base_vals:
        base_vals["sex_est"] = median_sex

    # Tooth class reference is the dropped category (e.g., Anterior)
    # so all tooth_class_* dummies remain 0 here.
    x_nonhuman = np.array([base_vals[col] for col in exog.columns], dtype=float)
    x_human = x_nonhuman.copy()
    human_idx = list(exog.columns).index("is_human")
    x_human[human_idx] = 1.0

    logit_nonhuman = float(np.dot(x_nonhuman, result.params))
    logit_human = float(np.dot(x_human, result.params))
    p_nonhuman = float(1 / (1 + np.exp(-logit_nonhuman)))
    p_human_pred = float(1 / (1 + np.exp(-logit_human)))
    diff = p_human_pred - p_nonhuman

    # Map evidence to Likert-scale response (0–100)
    # Strength from p-value (0 at p>=0.05, 1 at p=0)
    p_strength = max(0.0, min(1.0, 1.0 - p_human / 0.05))
    # Strength from effect size: cap at 0.1 absolute difference
    d_strength = max(0.0, min(1.0, abs(diff) / 0.1)) if not np.isnan(diff) else 0.0
    strength = 0.5 * p_strength + 0.5 * d_strength

    if p_human < 0.05 and diff > 0:
        # Yes: humans have higher AMTL after accounting for covariates
        response = int(round(50 + 50 * strength))
        yes_no_text = "Yes"
    else:
        # No (either not significant or effect is in the opposite direction)
        response = int(round(50 - 50 * strength))
        yes_no_text = "No"

    response = max(0, min(100, response))

    explanation_lines = [
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio), after accounting for age, sex, and tooth class?",
        "",
        "Data and model:",
        "- Dataset includes 1,450 rows with counts of missing teeth and observable sockets by specimen and tooth class.",
        "- Response modeled as a binomial outcome: number of missing teeth out of observable sockets for each row.",
        "- Predictors: human vs non-human indicator, age at death (years), estimated sex, and tooth class (anterior vs posterior/premolar).",
        "",
        "Key model results for the human indicator (Homo sapiens vs non-human primates):",
        f"- Logistic regression coefficient for humans: {coef_human:.3f} (SE {se_human:.3f}, p-value {p_human:.3g}).",
        f"- Corresponding odds ratio for AMTL in humans vs non-human primates: {odds_ratio_human:.2f}.",
        "",
        "Predicted AMTL probabilities at typical covariate values (median age and sex, anterior teeth):",
        f"- Non-human primates: predicted probability of a tooth being missing = {p_nonhuman:.3f}.",
        f"- Modern humans: predicted probability of a tooth being missing = {p_human_pred:.3f}.",
        f"- Difference (humans minus non-humans): {diff:.3f}.",
        "",
        f"Interpretation:",
        f"- Direction of effect (humans {'>' if diff > 0 else '<='} non-humans) and p-value indicate: {yes_no_text} to the question of whether humans have higher AMTL frequencies after controlling for age, sex, and tooth class.",
        f"- The Likert-scale response of {response} (0 = strong 'No', 100 = strong 'Yes') reflects both the statistical significance (p-value) and the magnitude of the predicted difference in AMTL probability between humans and non-human primates.",
    ]

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()

