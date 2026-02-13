import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_analysis():
    """Load data, clean it, and fit a binomial regression model."""
    df = pd.read_csv("amtl.csv")

    # Basic validity filters
    df = df.copy()
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]
    df = df[df["num_amtl"] <= df["sockets"]]
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "tooth_class",
            "genus",
        ]
    )

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = df["genus"].str.contains("Homo", case=False, na=False).astype(int)

    # Genus-level descriptive summary (overall AMTL proportions)
    genus_summary = (
        df.groupby("genus")
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .reset_index()
    )
    genus_summary["prop_missing"] = (
        genus_summary["total_missing"] / genus_summary["total_sockets"]
    )

    # Binomial regression: successes vs. failures, with covariates
    X = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    )
    X = sm.add_constant(X, has_constant="add")

    y = np.column_stack(
        [
            df["num_amtl"].to_numpy(),
            (df["sockets"] - df["num_amtl"]).to_numpy(),
        ]
    )

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    coef = float(result.params["is_human"])
    pvalue = float(result.pvalues["is_human"])

    # Predicted probabilities for a typical specimen (average age/sex, baseline tooth class)
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())

    columns = list(X.columns)
    base_row = {col: 0.0 for col in columns}
    if "const" in base_row:
        base_row["const"] = 1.0

    base_row["age"] = mean_age
    base_row["prob_male"] = mean_prob_male

    rows = []
    for is_h in (0, 1):
        row = dict(base_row)
        row["is_human"] = float(is_h)
        rows.append(row)

    X_pred = pd.DataFrame(rows)[columns]
    pred_probs = result.predict(X_pred)
    prob_nonhuman = float(pred_probs.iloc[0])
    prob_human = float(pred_probs.iloc[1])

    genus_props = {
        row["genus"]: float(row["prop_missing"]) for _, row in genus_summary.iterrows()
    }

    analysis = {
        "coef_is_human": coef,
        "pvalue_is_human": pvalue,
        "prob_nonhuman_typical": prob_nonhuman,
        "prob_human_typical": prob_human,
        "genus_prop_missing": genus_props,
        "n_rows": int(df.shape[0]),
    }

    return analysis


def choose_response(analysis):
    """Map model results to a 0–100 Likert response and explanation."""
    coef = analysis["coef_is_human"]
    pvalue = analysis["pvalue_is_human"]
    prob_human = analysis["prob_human_typical"]
    prob_nonhuman = analysis["prob_nonhuman_typical"]

    # Choose response strength based on sign and significance of human effect
    if coef < 0 and pvalue < 0.01:
        response = 5
    elif coef < 0 and pvalue < 0.05:
        response = 15
    elif coef < 0 and pvalue < 0.1:
        response = 30
    elif abs(coef) < 0.05 or pvalue > 0.5:
        response = 40 if coef <= 0 else 60
    elif coef > 0 and pvalue < 0.01:
        response = 95
    elif coef > 0 and pvalue < 0.05:
        response = 85
    elif coef > 0 and pvalue < 0.1:
        response = 70
    else:
        response = 55 if coef > 0 else 45

    genus_lines = [
        f"{genus}: {prop:.3f}"
        for genus, prop in sorted(analysis["genus_prop_missing"].items())
    ]

    direction = "lower" if coef < 0 else "higher"
    supports_claim = coef > 0 and pvalue < 0.05

    explanation = (
        "I analyzed the antemortem tooth loss (AMTL) dataset using a binomial logistic "
        "regression where the outcome was the number of missing teeth out of the total "
        "observable sockets, modeled with predictors for whether the specimen is a modern "
        "human versus a non-human primate, age at death, estimated probability of being "
        "male, and tooth class (anterior, posterior, premolar).\n"
        f"The estimated coefficient for humans versus non-humans was {coef:.3f} "
        f"(p = {pvalue:.3g}), indicating {direction} odds of AMTL for modern humans "
        "after accounting for age, sex, and tooth class.\n"
        f"For a typical specimen (average age and sex), the model predicts AMTL "
        f"probabilities of {prob_nonhuman:.3f} for non-human primates and "
        f"{prob_human:.3f} for humans.\n"
        "Aggregating the raw data by genus, the overall AMTL proportions "
        "(total missing teeth divided by total observable sockets) are: "
        + "; ".join(genus_lines)
        + ".\n"
        + (
            "Given that the adjusted human effect is higher and the predicted human "
            "probability is meaningfully above that of non-human primates, these data "
            "support the claim that modern humans have higher frequencies of AMTL than "
            "Pan, Pongo, and Papio."
            if supports_claim
            else "Given that the adjusted human effect is not higher (or is even lower) "
            "and the predicted human probability is not meaningfully above that of "
            "non-human primates, these data do not support the claim that modern humans "
            "have higher frequencies of AMTL than Pan, Pongo, and Papio."
        )
        + f" Accordingly, I assign a Likert-scale response of {response} "
        "(0 = strong 'No', 100 = strong 'Yes')."
    )

    return int(response), explanation


def main():
    analysis = run_analysis()
    response, explanation = choose_response(analysis)

    output = {"response": int(response), "explanation": explanation}
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    # Also print a readable summary for interactive inspection
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
