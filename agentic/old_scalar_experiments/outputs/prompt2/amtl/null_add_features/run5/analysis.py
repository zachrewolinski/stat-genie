import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only rows with valid socket counts and key variables present
    df = df[df["sockets"] > 0].copy()
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )
    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    # AMTL proportion per tooth class for each specimen
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]
    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression: AMTL proportion with sockets as trial weights
    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_rates(df: pd.DataFrame) -> pd.Series:
    # Mean AMTL rate per socket by genus (descriptive)
    return df.groupby("genus", observed=True)["amtl_prop"].mean()


def decide_answer(result, genus_rates: pd.Series):
    coef = result.params.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)

    # Direction and significance of human effect in the model
    human_higher = bool(coef > 0)
    statistically_significant = bool(pval < 0.05) if np.isfinite(pval) else False

    # Descriptive comparison: human vs non-human mean rates
    human_rate = genus_rates.get("Homo sapiens", np.nan)
    non_human_rates = genus_rates.drop(labels=["Homo sapiens"], errors="ignore")
    non_human_mean = non_human_rates.mean() if len(non_human_rates) > 0 else np.nan

    descriptively_higher = (
        bool(human_rate > non_human_mean)
        if np.isfinite(human_rate) and np.isfinite(non_human_mean)
        else False
    )

    if human_higher and (statistically_significant or descriptively_higher):
        response = "Yes"
    else:
        response = "No"

    # Map p-value and consistency between model and descriptives to a confidence score
    if not np.isfinite(pval):
        base_conf = 50
    elif pval < 0.001:
        base_conf = 95
    elif pval < 0.01:
        base_conf = 90
    elif pval < 0.05:
        base_conf = 80
    elif pval < 0.1:
        base_conf = 70
    else:
        base_conf = 60

    # Adjust confidence downward if model and descriptive directions disagree
    if human_higher != descriptively_higher:
        base_conf -= 10

    # If we are answering "No", interpret confidence as confidence that humans
    # do NOT have higher AMTL after accounting for covariates.
    confidence = int(max(0, min(100, base_conf)))

    return response, confidence, {
        "coef_is_human": float(coef) if np.isfinite(coef) else None,
        "pval_is_human": float(pval) if np.isfinite(pval) else None,
        "human_rate": float(human_rate) if np.isfinite(human_rate) else None,
        "non_human_mean_rate": float(non_human_mean)
        if np.isfinite(non_human_mean)
        else None,
    }


def build_explanation(
    response: str, confidence: int, result, genus_rates: pd.Series, stats_summary: dict
) -> str:
    lines = []
    lines.append(
        "I modeled the frequency of antemortem tooth loss (AMTL) per tooth socket "
        "using binomial regression with a logit link."
    )
    lines.append(
        "The outcome was the proportion of missing teeth (`num_amtl / sockets`) for "
        "each specimen–tooth-class combination, with the number of sockets used as "
        "binomial trial weights."
    )
    lines.append(
        "Predictors included an indicator for modern humans vs non-human primates "
        "(`is_human`), age at death, probability of being male (`prob_male`), and "
        "tooth class (anterior, posterior, premolar)."
    )

    coef = stats_summary["coef_is_human"]
    pval = stats_summary["pval_is_human"]
    human_rate = stats_summary["human_rate"]
    non_human_mean = stats_summary["non_human_mean_rate"]

    if coef is not None and pval is not None:
        lines.append(
            f"In the regression, the coefficient for modern humans (`is_human`) "
            f"was {coef:.3f} on the log-odds scale with p-value {pval:.4f}."
        )

    if human_rate is not None and non_human_mean is not None:
        lines.append(
            "Descriptively, the mean AMTL proportion per socket was "
            f"{human_rate:.3f} for modern humans and "
            f"{non_human_mean:.3f} on average across non-human primate genera."
        )

    if response == "Yes":
        lines.append(
            "Both the direction of the human coefficient and the descriptive "
            "rates indicate that modern humans have higher AMTL frequencies than "
            "non-human primates after adjusting for age, sex, and tooth class."
        )
    else:
        lines.append(
            "Taken together, the regression results and descriptive rates do not "
            "support the claim that modern humans have higher AMTL frequencies "
            "than non-human primates once age, sex, and tooth class are accounted for."
        )

    lines.append(f"Based on this evidence, I assign a confidence of {confidence} "
                 f"out of 100 to the answer \"{response}\".")

    return " ".join(lines)


def main():
    df = load_data("amtl.csv")
    result = fit_model(df)
    genus_rates = summarize_genus_rates(df)
    response, confidence, stats_summary = decide_answer(result, genus_rates)
    explanation = build_explanation(response, confidence, result, genus_rates, stats_summary)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

