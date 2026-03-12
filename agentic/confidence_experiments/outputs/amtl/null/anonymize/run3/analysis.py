import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure numeric columns are properly typed
    num_cols = ["feature3", "feature4", "feature5", "feature6", "feature7"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows with missing critical values
    df = df.dropna(subset=["feature3", "feature4", "feature5", "feature7", "feature1", "feature8"])

    # Keep only rows with a positive number of observable sockets
    df = df[df["feature4"] > 0]

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Proportion of missing teeth within tooth class
    df = df.copy()
    df["missing_prop"] = df["feature3"] / df["feature4"]

    # Define human vs. non-human primate indicator based on genus field
    genus_values = df["feature8"].unique()
    # In this anonymized dataset, modern humans are labeled as "Homo sapiens"
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Treat tooth class as categorical
    df["feature1"] = df["feature1"].astype("category")

    # Center and scale continuous covariates mildly for numerical stability (not strictly necessary)
    for col in ["feature5", "feature7"]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df[f"{col}_c"] = (df[col] - mean_val) / std_val
        else:
            df[f"{col}_c"] = df[col] - mean_val

    return df


def fit_binomial_glm(df: pd.DataFrame):
    # Binomial regression on proportion with number of trials as frequency weights
    formula = "missing_prop ~ is_human + feature5_c + feature7_c + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()
    return result


def compute_marginal_difference(df: pd.DataFrame, result) -> dict:
    # Average marginal effect of being human vs non-human:
    # predict for each observation once as human and once as non-human, holding other variables fixed.
    base = df.copy()

    human = base.copy()
    human["is_human"] = 1
    nonhuman = base.copy()
    nonhuman["is_human"] = 0

    pred_human = result.predict(human)
    pred_nonhuman = result.predict(nonhuman)

    avg_human = float(pred_human.mean())
    avg_nonhuman = float(pred_nonhuman.mean())
    diff = avg_human - avg_nonhuman

    return {
        "avg_pred_missing_human": avg_human,
        "avg_pred_missing_nonhuman": avg_nonhuman,
        "avg_difference_human_minus_nonhuman": diff,
    }


def derive_likert_score(coef: float, pvalue: float, diff: float) -> int:
    """
    Map statistical evidence into a 0-100 Likert score where higher means stronger evidence
    that modern humans have higher AMTL rates than non-human primates.
    """
    # If the direction is opposite (humans lower AMTL), lean strongly "No"
    if coef < 0 or diff < 0:
        if pvalue < 0.05:
            return 5  # strong evidence against humans having higher AMTL
        if pvalue < 0.1:
            return 10
        return 20

    # Direction consistent with hypothesis: humans higher AMTL
    if pvalue < 0.001:
        # Very strong evidence; scale with effect size (difference in proportions)
        if diff > 0.10:
            return 95
        if diff > 0.05:
            return 90
        return 85
    if pvalue < 0.01:
        if diff > 0.10:
            return 85
        if diff > 0.05:
            return 80
        return 75
    if pvalue < 0.05:
        if diff > 0.05:
            return 70
        return 65
    if pvalue < 0.1:
        return 55  # weak, marginal evidence in the expected direction

    # Little to no statistical support
    return 40


def build_explanation(result, marginal_stats: dict, likert_score: int) -> str:
    coef = float(result.params["is_human"])
    pvalue = float(result.pvalues["is_human"])
    avg_h = marginal_stats["avg_pred_missing_human"]
    avg_nh = marginal_stats["avg_pred_missing_nonhuman"]
    diff = marginal_stats["avg_difference_human_minus_nonhuman"]

    explanation = (
        "Using the AMTL dataset (n = {n}), I modeled the proportion of missing teeth "
        "within each tooth class as a binomial outcome (number of missing teeth out of "
        "the number of observable sockets) using a generalized linear model with a "
        "logit link. The predictors included an indicator for modern humans versus "
        "non-human primates, estimated age at death, estimated sex, and tooth class "
        "(anterior, posterior, premolar). Age and sex were treated as continuous "
        "covariates, and tooth class as a categorical factor. "
        "In this model, the coefficient for the human indicator was {coef:.3f} with a "
        "p-value of {pvalue:.3g}, indicating that, after adjusting for age, sex, and "
        "tooth class, modern humans have {direction} AMTL rates than the pooled "
        "non-human primate genera (Pan, Pongo, Papio). "
        "Based on model-based marginal predictions, the average predicted proportion of "
        "missing teeth is approximately {avg_h:.3f} for modern humans and {avg_nh:.3f} "
        "for non-human primates, a difference of {diff:.3f} (human minus non-human). "
        "Taken together, these results provide {strength} statistical evidence that "
        "modern humans {qualifier} higher frequencies of antemortem tooth loss than "
        "non-human primates once age, sex, and tooth class are accounted for. "
        "I therefore translate this conclusion into a Likert-scale response of "
        "{score:d} on a 0–100 scale, where values near 0 represent a strong 'No' and "
        "values near 100 represent a strong 'Yes' to the research question."
    )

    direction = "higher" if coef > 0 else "lower"
    if pvalue < 0.001:
        strength = "very strong"
    elif pvalue < 0.01:
        strength = "strong"
    elif pvalue < 0.05:
        strength = "moderate"
    elif pvalue < 0.1:
        strength = "weak"
    else:
        strength = "little to no"

    qualifier = "do" if coef > 0 else "do not"

    return explanation.format(
        n=len(result.model.endog),
        coef=coef,
        pvalue=pvalue,
        direction=direction,
        avg_h=avg_h,
        avg_nh=avg_nh,
        diff=diff,
        strength=strength,
        qualifier=qualifier,
        score=likert_score,
    )


def main():
    csv_path = Path("amtl.csv")
    df_raw = load_data(csv_path)
    df = prepare_data(df_raw)

    # Fit model
    result = fit_binomial_glm(df)

    # Extract key statistics for the human indicator
    coef_human = float(result.params["is_human"])
    pvalue_human = float(result.pvalues["is_human"])

    # Compute marginal predicted differences
    marginal_stats = compute_marginal_difference(df, result)
    diff = marginal_stats["avg_difference_human_minus_nonhuman"]

    # Map results to Likert score
    likert_score = derive_likert_score(coef_human, pvalue_human, diff)

    # Build narrative explanation
    explanation = build_explanation(result, marginal_stats, likert_score)

    # Save conclusion as JSON
    conclusion = {"response": int(likert_score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

