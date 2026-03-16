import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    expected_cols = {
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "genus",
        "pop",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def fit_binomial_model(df: pd.DataFrame):
    # Create binary indicator for modern humans vs non-human primates
    df = df.copy()
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth and number of trials
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Drop any rows with non-finite values just in case
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prop_amtl", "age", "prob_male", "tooth_class", "is_human", "sockets"]
    )

    # Design matrix: intercept, human indicator, age, sex, tooth class (categorical)
    X = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
        dtype=float,
    )
    X = sm.add_constant(X, has_constant="add")

    y = df["prop_amtl"]
    weights = df["sockets"]

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()
    return result, df, X


def summarize_results(result, df: pd.DataFrame, X: pd.DataFrame):
    # Extract human coefficient and statistics
    coef_human = result.params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    pval_human = result.pvalues.get("is_human", np.nan)

    odds_ratio_human = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    # Predicted probabilities at average covariate values for humans vs non-humans
    mean_row = X.mean(axis=0)

    mean_row_human = mean_row.copy()
    mean_row_human["is_human"] = 1.0

    mean_row_nonhuman = mean_row.copy()
    mean_row_nonhuman["is_human"] = 0.0

    # Linear predictors and probabilities
    lp_human = float(np.dot(result.params, mean_row_human))
    lp_nonhuman = float(np.dot(result.params, mean_row_nonhuman))

    prob_human = float(1.0 / (1.0 + np.exp(-lp_human)))
    prob_nonhuman = float(1.0 / (1.0 + np.exp(-lp_nonhuman)))

    # Map evidence strength to Likert-style 0–100 scale
    # We treat p-value and effect size jointly.
    if np.isnan(pval_human) or np.isnan(coef_human):
        response_score = 50
    else:
        if coef_human > 0 and pval_human < 0.001:
            response_score = 95
        elif coef_human > 0 and pval_human < 0.01:
            response_score = 85
        elif coef_human > 0 and pval_human < 0.05:
            response_score = 75
        elif coef_human > 0:
            response_score = 60
        elif coef_human < 0 and pval_human < 0.05:
            response_score = 20
        elif coef_human < 0:
            response_score = 40
        else:
            response_score = 50

    response_score = int(round(response_score))

    # Build explanation string
    genus_summary = (
        df.assign(
            total_missing=df["num_amtl"],
            total_sockets=df["sockets"],
        )
        .groupby("genus")[["total_missing", "total_sockets"]]
        .sum()
    )
    genus_summary["amtl_rate"] = genus_summary["total_missing"] / genus_summary["total_sockets"]

    # Format genus-level rates
    genus_lines = []
    for genus, row in genus_summary.iterrows():
        rate = row["amtl_rate"]
        genus_lines.append(f"- {genus}: {rate:.3f} proportion of teeth missing")
    genus_text = "\n".join(genus_lines)

    # Interpret direction and significance of the human effect
    if coef_human > 0 and pval_human < 0.05:
        qualitative_conclusion = (
            "the data provide statistically significant evidence that modern humans have higher "
            "frequencies of antemortem tooth loss than the non-human primate genera considered. "
            "I therefore answer 'Yes' to the research question."
        )
    elif coef_human < 0 and pval_human < 0.05:
        qualitative_conclusion = (
            "the data provide statistically significant evidence that modern humans have lower "
            "frequencies of antemortem tooth loss than the non-human primate genera considered. "
            "Because the effect is in the opposite direction, I answer 'No' to the research question "
            "of whether modern humans have higher AMTL frequencies."
        )
    else:
        qualitative_conclusion = (
            "the estimated difference between modern humans and non-human primates in AMTL frequency "
            "is small and not statistically significant. The model does not provide strong evidence "
            "that modern humans have higher AMTL frequencies than non-human primates, so I answer "
            "'No' to the research question."
        )

    explanation = (
        "I fit a binomial regression model for the proportion of antemortem tooth loss "
        "(num_amtl / sockets) using a logit link, with each row weighted by the number of "
        "observable tooth sockets. The predictors included a binary indicator for modern humans "
        "(Homo sapiens vs. all non-human primates), age at death, estimated sex (prob_male), "
        "and tooth class (with dummy variables).\n\n"
        f"The coefficient for the human indicator (Homo sapiens vs non-human primates) was "
        f"{coef_human:.3f} on the log-odds scale, corresponding to an odds ratio of "
        f"{odds_ratio_human:.2f}, with a p-value of {pval_human:.4g}. "
        f"At average covariate values, the model-estimated probability that a given tooth "
        f"is missing was {prob_human:.3f} for modern humans and {prob_nonhuman:.3f} for "
        "non-human primates.\n\n"
        "Genus-level raw AMTL rates (total missing teeth divided by total observable sockets) were:\n"
        f"{genus_text}\n\n"
        "Taken together, "
        f"{qualitative_conclusion} "
        f"This assessment is summarized by a Likert-style response score of {response_score} "
        "on a 0–100 scale, where higher values indicate stronger evidence for a 'Yes' answer."
    )

    return response_score, explanation


def write_conclusion(path: Path, response_score: int, explanation: str) -> None:
    obj = {
        "response": int(response_score),
        "explanation": explanation,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False))


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "amtl.csv"
    conclusion_path = base_dir / "conclusion.txt"

    df = load_data(csv_path)
    result, df_used, X = fit_binomial_model(df)
    response_score, explanation = summarize_results(result, df_used, X)
    write_conclusion(conclusion_path, response_score, explanation)


if __name__ == "__main__":
    main()
