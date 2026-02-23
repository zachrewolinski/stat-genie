import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str) -> pd.DataFrame:
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


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """Expand row-level counts (num_amtl out of sockets) to tooth-level binary outcomes."""
    records = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        missing = int(row["num_amtl"])
        present = sockets - missing
        if sockets <= 0:
            continue

        outcomes = np.array([1] * missing + [0] * present, dtype=int)
        for y in outcomes:
            records.append(
                {
                    "amtl": y,
                    "age": float(row["age"]),
                    "prob_male": float(row["prob_male"]),
                    "tooth_class": row["tooth_class"],
                    "genus": row["genus"],
                    "specimen": row["specimen"],
                    "pop": row["pop"],
                }
            )

    tooth_df = pd.DataFrame.from_records(records)
    # Indicator for modern humans vs non-human primates
    tooth_df["human"] = (tooth_df["genus"] == "Homo sapiens").astype(int)
    return tooth_df


def fit_binomial_model(tooth_df: pd.DataFrame):
    """Fit a logistic regression with cluster-robust SEs by specimen."""
    formula = "amtl ~ human + age + prob_male + C(tooth_class)"
    model = smf.glm(formula=formula, data=tooth_df, family=sm.families.Binomial())
    result = model.fit(cov_type="cluster", cov_kwds={"groups": tooth_df["specimen"]})
    return result


def summarize_effect(df: pd.DataFrame, tooth_df: pd.DataFrame, result) -> dict:
    # Observed AMTL proportions by genus
    genus_summary = (
        df.groupby("genus")
        .apply(lambda g: g["num_amtl"].sum() / g["sockets"].sum())
        .sort_values(ascending=False)
    )

    beta_human = result.params.get("human", np.nan)
    se_human = result.bse.get("human", np.nan)
    p_human = result.pvalues.get("human", np.nan)

    ci = result.conf_int().loc["human"]
    ci_lower, ci_upper = ci[0], ci[1]

    or_human = float(np.exp(beta_human))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Predicted probabilities for a typical case (median age, median prob_male, Posterior teeth)
    median_age = float(df["age"].median())
    median_prob_male = float(df["prob_male"].median())
    typical = pd.DataFrame(
        {
            "human": [0, 1],
            "age": [median_age, median_age],
            "prob_male": [median_prob_male, median_prob_male],
            "tooth_class": ["Posterior", "Posterior"],
        }
    )
    pred_probs = result.predict(typical)
    prob_nonhuman = float(pred_probs.iloc[0])
    prob_human = float(pred_probs.iloc[1])

    # Map evidence to a 0–100 Likert-style response
    # Use both statistical significance and effect size (odds ratio)
    # Base strength from p-value
    if np.isnan(p_human) or np.isnan(or_human):
        response = 50
        qualitative = "inconclusive"
    else:
        abs_log_or = float(abs(np.log(or_human)))
        if abs_log_or < 0.05:
            # Effect size is essentially null regardless of p-value
            base_strength = 50
        elif p_human < 0.001:
            base_strength = 95
        elif p_human < 0.01:
            base_strength = 85
        elif p_human < 0.05:
            base_strength = 75
        elif p_human < 0.1:
            base_strength = 60
        else:
            base_strength = 50

        if or_human > 1.0 and base_strength > 50:
            response = int(round(base_strength))
            qualitative = "yes"
        elif or_human < 1.0 and base_strength > 50:
            response = int(round(100 - base_strength))
            qualitative = "no"
        else:
            response = 50
            qualitative = "inconclusive"

    explanation_lines = []
    explanation_lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher "
        "frequencies of antemortem tooth loss (AMTL) than non-human primates "
        "(Pan, Pongo, Papio) after accounting for age, sex, and tooth class?"
    )
    explanation_lines.append(
        "Data and outcome: The dataset contains counts of missing teeth "
        "(`num_amtl`) and observable sockets (`sockets`) for different tooth "
        "classes (anterior, posterior, premolar) within each specimen. "
        "I expanded these counts to the tooth level so that each tooth is "
        "coded as present (0) or missing (1)."
    )
    explanation_lines.append(
        "Modeling approach: I fit a binomial logistic regression at the tooth "
        "level with AMTL (missing vs present) as the outcome, and predictors "
        "including a binary indicator for modern humans vs non-human primates "
        "(`human`), estimated age at death (`age`), probability of being male "
        "(`prob_male`), and categorical tooth class. Cluster-robust standard "
        "errors were used to account for multiple teeth from the same specimen."
    )
    explanation_lines.append(
        "Observed AMTL proportions by genus (num_amtl / sockets) show that "
        f"humans and non-human primates differ in their raw frequencies: "
        + ", ".join(
            f"{genus}: {prop:.3f}"
            for genus, prop in genus_summary.items()
        )
        + "."
    )
    explanation_lines.append(
        "Regression results: The human indicator has an odds ratio of "
        f"{or_human:.2f} (95% CI {or_ci_lower:.2f}–{or_ci_upper:.2f}, "
        f"p = {p_human:.4g}). Values above 1 indicate higher odds of AMTL "
        "for modern humans relative to non-human primates, after adjusting "
        "for age, sex, and tooth class."
    )
    explanation_lines.append(
        "For a typical individual (median age and sex probability, posterior "
        f"teeth), the model predicts an AMTL probability of "
        f"{prob_nonhuman:.3f} for non-human primates vs "
        f"{prob_human:.3f} for modern humans."
    )

    if qualitative == "yes":
        explanation_lines.append(
            "Interpretation: The odds ratio is greater than 1 and statistically "
            "significant, providing evidence that modern humans experience "
            "higher frequencies of AMTL than non-human primates after "
            "controlling for age, sex, and tooth class."
        )
    elif qualitative == "no":
        explanation_lines.append(
            "Interpretation: The odds ratio is less than 1 and statistically "
            "significant, indicating that modern humans do not have higher "
            "frequencies of AMTL than non-human primates; if anything, they "
            "show lower AMTL after controlling for age, sex, and tooth class."
        )
    else:
        explanation_lines.append(
            "Interpretation: The estimated effect of being human vs non-human "
            "is small and/or not statistically significant, so the data do not "
            "provide clear evidence that modern humans differ in AMTL "
            "frequency from non-human primates after adjustment."
        )

    explanation_lines.append(
        f"Likert-scale conclusion (0 = strong 'No', 100 = strong 'Yes'): "
        f"{response}. This value reflects the combination of effect size and "
        "statistical significance for the human vs non-human contrast."
    )

    explanation = "\n\n".join(explanation_lines)

    return {
        "response": int(response),
        "explanation": explanation,
    }


def main():
    csv_path = "amtl.csv"
    df = load_data(csv_path)
    tooth_df = expand_to_tooth_level(df)
    result = fit_binomial_model(tooth_df)
    summary = summarize_effect(df, tooth_df, result)

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w") as f:
        json.dump(
            {
                "response": summary["response"],
                "explanation": summary["explanation"],
            },
            f,
        )


if __name__ == "__main__":
    main()

