import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean the AMTL dataset."""
    df = pd.read_csv(csv_path)

    # Rename columns to reflect their semantic meaning based on info.json
    df = df.rename(
        columns={
            "sockets": "tooth_class",  # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",  # Unique specimen identifier
            "genus": "n_missing",  # Number of missing teeth of that class
            "age": "n_sockets",  # Number of observable sockets
            "pop": "age_at_death",  # Estimated age at death
            "num_amtl": "age_uncertainty",  # Uncertainty in age estimate
            "stdev_age": "sex_estimate",  # Estimate / probability of male
            "tooth_class": "genus",  # Homo sapiens, Pan, Papio, Pongo
            "specimen": "population",  # Region / population
        }
    )

    # Basic sanity checks to avoid invalid rows in the model
    df = df[(df["n_sockets"] > 0) & (df["n_missing"] >= 0)]
    df = df[df["n_missing"] <= df["n_sockets"]]

    # Proportion of missing teeth per row
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Categorical tooth class (anterior/posterior/premolar)
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Center age at death for numerical stability
    df["age_at_death_c"] = df["age_at_death"] - df["age_at_death"].mean()

    return df


def fit_model(df: pd.DataFrame):
    """Fit a binomial regression model for AMTL frequency."""
    # Binomial GLM with logit link; use n_sockets as number of trials
    formula = "prop_missing ~ is_human + age_at_death_c + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(result, df: pd.DataFrame):
    """Extract human vs non-human effect and compute a Likert score."""
    params = result.params
    pvalues = result.pvalues

    # Effect of being human on log-odds of AMTL
    coef_human = params.get("is_human", np.nan)
    p_human = pvalues.get("is_human", np.nan)
    or_human = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    # Construct a Likert score based on effect size and significance
    if not np.isfinite(coef_human) or not np.isfinite(p_human):
        likert = 50
    else:
        if p_human < 0.001:
            base = 90
        elif p_human < 0.01:
            base = 80
        elif p_human < 0.05:
            base = 70
        elif p_human < 0.1:
            base = 60
        else:
            base = 50

        # Adjust score modestly by effect size (odds ratio)
        if or_human > 1:
            likert = min(100, int(round(base + min(10, (or_human - 1) * 5))))
        else:
            likert = max(0, int(round(base - min(10, (1 - or_human) * 5))))

    # Predicted probabilities for a typical specimen
    median_age_c = 0.0  # because of centering
    median_sex = df["sex_estimate"].median()
    # Use the most frequent tooth class as reference scenario
    mode_tooth_class = df["tooth_class"].mode().iat[0]

    design = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_at_death_c": [median_age_c, median_age_c],
            "sex_estimate": [median_sex, median_sex],
            "tooth_class": [mode_tooth_class, mode_tooth_class],
        }
    )

    preds = result.get_prediction(design).summary_frame()
    prob_nonhuman = float(preds["mean"].iloc[0])
    prob_human = float(preds["mean"].iloc[1])

    return {
        "coef_human": float(coef_human),
        "p_human": float(p_human),
        "or_human": or_human,
        "prob_nonhuman": prob_nonhuman,
        "prob_human": prob_human,
        "likert": likert,
    }


def build_explanation(summary: dict) -> str:
    """Create a textual explanation of the findings."""
    coef = summary["coef_human"]
    pval = summary["p_human"]
    or_human = summary["or_human"]
    prob_nonhuman = summary["prob_nonhuman"]
    prob_human = summary["prob_human"]

    direction = "higher" if or_human > 1 else "lower" if or_human < 1 else "similar"

    explanation = (
        "I fit a binomial regression model for the proportion of missing teeth "
        "(number of missing teeth out of the number of observable sockets) using "
        "a logit link. The model included an indicator for modern humans "
        "(Homo sapiens vs. non-human primates), estimated age at death, an "
        "estimate of sex, and tooth class (anterior, posterior, premolar) as "
        "predictors.\n\n"
        f"The coefficient for being human on the log-odds scale was {coef:.3f}, "
        f"corresponding to an odds ratio of approximately {or_human:.2f}, with a "
        f"p-value of {pval:.3g}. For a typical specimen (median age, median sex "
        f"estimate, and the most common tooth class), the model predicts an AMTL "
        f"probability of about {prob_nonhuman:.3f} for non-human primates and "
        f"{prob_human:.3f} for humans. This indicates that humans have {direction} "
        "frequencies of antemortem tooth loss compared to non-human primates after "
        "adjusting for age, sex, and tooth class.\n\n"
        "Based on the magnitude of the estimated odds ratio and its statistical "
        "significance, I translate this into a Likert-scale assessment of how "
        "strongly the data support the claim that modern humans have higher AMTL "
        "frequencies than non-human primates."
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    result = fit_model(df)
    summary = summarize_effect(result, df)

    explanation = build_explanation(summary)

    conclusion = {
        "response": int(summary["likert"]),
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

