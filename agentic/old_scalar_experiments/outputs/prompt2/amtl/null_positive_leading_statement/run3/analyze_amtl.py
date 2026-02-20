import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only rows with valid socket counts
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]

    # Focus on the four genera of interest
    genera_of_interest = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Define human vs non-human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth in that tooth class for the specimen
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Ensure categorical variables are treated as such
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression using proportions with frequency weights
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    # Extract coefficient and p-value for human vs non-human
    coef = result.params.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)

    # Average predicted AMTL probability per tooth for humans vs non-humans
    df = df.copy()
    df["predicted"] = result.predict(df)
    human_mean = df.loc[df["is_human"] == 1, "predicted"].mean()
    nonhuman_mean = df.loc[df["is_human"] == 0, "predicted"].mean()

    rel_diff = (human_mean - nonhuman_mean) / nonhuman_mean if nonhuman_mean > 0 else np.nan

    return {
        "coef_is_human": float(coef),
        "p_is_human": float(pval),
        "human_mean": float(human_mean),
        "nonhuman_mean": float(nonhuman_mean),
        "relative_difference": float(rel_diff),
    }


def make_conclusion(stats: dict) -> dict:
    coef = stats["coef_is_human"]
    pval = stats["p_is_human"]
    human_mean = stats["human_mean"]
    nonhuman_mean = stats["nonhuman_mean"]
    rel_diff = stats["relative_difference"]

    # Heuristic decision rule:
    # - Direction: humans show higher predicted AMTL (coef > 0 and human_mean > nonhuman_mean)
    # - Strength: p-value < 0.05 and relative difference at least 10%
    direction_support = coef > 0 and human_mean > nonhuman_mean
    significant = pval < 0.05
    meaningful = rel_diff > 0.10

    if direction_support and significant and meaningful:
        response = "Yes"
        confidence = 85
    elif direction_support and (significant or meaningful):
        response = "Yes"
        confidence = 70
    elif direction_support:
        response = "Yes"
        confidence = 60
    else:
        response = "No"
        confidence = 70 if significant else 60

    explanation_parts = [
        "I fit a binomial regression model of the proportion of missing teeth (num_amtl / sockets) with predictors for human vs non-human primates (is_human), age at death, estimated sex (prob_male), and tooth class.",
        f"The coefficient for the human indicator was {coef:.3f} with p-value {pval:.3g}.",
        f"The model-based mean predicted AMTL probability per tooth was {human_mean:.3f} for humans and {nonhuman_mean:.3f} for non-human primates, a relative difference of approximately {rel_diff * 100:.1f}%.",
    ]

    if response == "Yes":
        explanation_parts.append(
            "Because humans show higher predicted AMTL frequencies than non-human primates after accounting for age, sex, and tooth class—and this difference is supported by the model coefficients—I conclude that humans have higher AMTL rates under this model."
        )
    else:
        explanation_parts.append(
            "Given the estimated effect size and statistical support, the data do not provide clear evidence that humans have higher AMTL frequencies than non-human primates once age, sex, and tooth class are accounted for."
        )

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main():
    df = load_data("amtl.csv")
    result = fit_model(df)

    print(result.summary())

    stats = summarize_effect(df, result)
    print("Summary statistics for human vs non-human effect:")
    print(json.dumps(stats, indent=2))

    conclusion = make_conclusion(stats)

    # Write conclusion.json-like output to conclusion.txt as required
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

