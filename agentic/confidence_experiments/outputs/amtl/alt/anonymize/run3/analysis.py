import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data() -> pd.DataFrame:
    """Load CSV and prepare columns according to metadata description."""
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature7": "sex",
            "feature8": "genus",
        }
    )

    # Create human indicator (modern humans vs. non-human primates)
    df["is_human"] = df["genus"].astype(str).str.contains("Homo", regex=False).astype(int)

    # Basic sanity: drop any rows with non-positive sockets (should not occur per metadata)
    df = df[df["n_sockets"] > 0].copy()

    return df


def summarize_by_genus(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Crude AMTL frequencies by genus."""
    grouped = (
        df.groupby("genus", as_index=True)
        .agg(total_missing=("n_missing", "sum"), total_sockets=("n_sockets", "sum"))
        .assign(missing_rate=lambda g: g["total_missing"] / g["total_sockets"])
    )

    summary: Dict[str, Dict[str, float]] = {}
    for genus, row in grouped.iterrows():
        summary[str(genus)] = {
            "total_missing": float(row["total_missing"]),
            "total_sockets": float(row["total_sockets"]),
            "missing_rate": float(row["missing_rate"]),
        }
    return summary


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand grouped counts (n_missing out of n_sockets) to tooth-level binary data.

    Each row in the original data becomes n_sockets rows with a binary AMTL outcome.
    This avoids having to rely on special binomial weighting semantics.
    """
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        n_missing = int(row["n_missing"])
        n_sockets = int(row["n_sockets"])
        n_present = n_sockets - n_missing

        base = {
            "is_human": int(row["is_human"]),
            "age": float(row["age"]),
            "sex": float(row["sex"]),
            "tooth_class": row["tooth_class"],
            "genus": row["genus"],
        }

        for _ in range(n_missing):
            rec = base.copy()
            rec["amtl"] = 1
            records.append(rec)

        for _ in range(n_present):
            rec = base.copy()
            rec["amtl"] = 0
            records.append(rec)

    long_df = pd.DataFrame.from_records(records)
    return long_df


def fit_logistic_model(long_df: pd.DataFrame):
    """
    Fit binomial regression for AMTL with human vs non-human indicator,
    adjusting for age, sex, and tooth class.
    """
    formula = "amtl ~ is_human + age + sex + C(tooth_class)"
    model = smf.glm(formula=formula, data=long_df, family=sm.families.Binomial())
    result = model.fit()
    return result


def compute_likert_score(coef: float, pvalue: float) -> int:
    """
    Map coefficient sign and p-value to a 0–100 Likert score.

    - Score > 50 => answer is effectively "Yes" to the research question.
    - Score < 50 => answer is effectively "No".
    """
    if np.isnan(coef) or np.isnan(pvalue):
        # Extremely defensive fallback: entirely uncertain.
        return 50

    if pvalue >= 0.05:
        # No statistically significant evidence either way.
        return 40

    if coef > 0:
        # Humans have higher AMTL odds than non-human primates.
        if pvalue < 0.001:
            return 95
        if pvalue < 0.01:
            return 90
        return 80

    # Humans have lower AMTL odds than non-human primates.
    if pvalue < 0.001:
        return 5
    if pvalue < 0.01:
        return 10
    return 20


def build_explanation(
    df: pd.DataFrame,
    long_df: pd.DataFrame,
    genus_summary: Dict[str, Dict[str, float]],
    coef: float,
    se: float,
    pvalue: float,
    odds_ratio: float,
    or_ci_low: float,
    or_ci_high: float,
    human_prob: float,
    nonhuman_prob: float,
    score: int,
) -> str:
    """Construct a human-readable explanation summarizing methods and findings."""
    n_obs = len(long_df)

    # Describe crude AMTL frequencies by genus.
    genus_parts: List[str] = []
    for genus, stats in genus_summary.items():
        rate_pct = stats["missing_rate"] * 100.0
        genus_parts.append(
            f"{genus}: {int(stats['total_missing'])}/{int(stats['total_sockets'])} teeth missing ({rate_pct:.1f}%)."
        )
    genus_text = " ".join(genus_parts)

    if pvalue >= 0.05:
        answer_word = "No"
        direction_sentence = (
            "There is no statistically significant evidence that modern humans differ "
            "from non-human primates in AMTL frequencies after adjusting for age, sex, and tooth class."
        )
    elif coef > 0:
        answer_word = "Yes"
        direction_sentence = (
            "Modern humans exhibit significantly higher odds of antemortem tooth loss (AMTL) than non-human primates "
            "of the same age, sex, and tooth class."
        )
    else:
        answer_word = "No"
        direction_sentence = (
            "Modern humans exhibit significantly lower odds of antemortem tooth loss (AMTL) than non-human primates "
            "of the same age, sex, and tooth class."
        )

    explanation = (
        f"{answer_word} – {direction_sentence} "
        f"I analyzed the AMTL dataset using binomial logistic regression at the individual-tooth level "
        f"({n_obs} tooth observations), where each tooth was coded as present or lost. "
        f"The model included a binary indicator for modern humans (Homo sapiens vs. non-human primates), "
        f"along with covariates for estimated age at death, estimated sex, and tooth class "
        f"(anterior, posterior, premolar). "
        f"The human indicator had coefficient {coef:.3f} (SE {se:.3f}, p={pvalue:.3g}), "
        f"corresponding to an odds ratio of {odds_ratio:.2f} "
        f"(95% CI {or_ci_low:.2f}–{or_ci_high:.2f}) for AMTL in humans relative to non-human primates "
        f"with the same age, sex, and tooth class. "
        f"Using the fitted model, at the average age and sex in the sample and for the most common tooth class, "
        f"the predicted probability of AMTL is {human_prob * 100:.1f}% for humans versus "
        f"{nonhuman_prob * 100:.1f}% for non-human primates. "
        f"Crude (unadjusted) AMTL frequencies by genus are: {genus_text} "
        f"Taken together, the direction, magnitude, and statistical significance of the human-vs-non-human coefficient "
        f"support a '{answer_word}' answer to the research question "
        f"\"Do modern humans have higher frequencies of AMTL than non-human primate genera (Pan, Pongo, Papio) "
        f"after accounting for age, sex, and tooth class?\". "
        f"On a 0–100 Likert scale where higher values represent stronger evidence that humans have higher AMTL "
        f"frequencies than non-human primates, I assign a score of {score}."
    )

    return explanation


def main() -> None:
    df = load_and_prepare_data()

    genus_summary = summarize_by_genus(df)

    # Expand to tooth-level observations for a straightforward binomial GLM.
    long_df = expand_to_tooth_level(df)

    # Fit logistic regression with human vs non-human indicator.
    result = fit_logistic_model(long_df)

    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    pvalue = float(result.pvalues["is_human"])

    # Odds ratio and 95% CI for the human vs non-human effect.
    ci_low, ci_high = result.conf_int().loc["is_human"]
    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Predicted probabilities at typical covariate values.
    mean_age = float(df["age"].mean())
    mean_sex = float(df["sex"].mean())
    common_tooth_class = df["tooth_class"].mode()[0]

    new_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "sex": [mean_sex, mean_sex],
            "tooth_class": [common_tooth_class, common_tooth_class],
        }
    )
    preds = result.predict(new_data)
    nonhuman_prob = float(preds.iloc[0])
    human_prob = float(preds.iloc[1])

    score = compute_likert_score(coef, pvalue)

    explanation = build_explanation(
        df=df,
        long_df=long_df,
        genus_summary=genus_summary,
        coef=coef,
        se=se,
        pvalue=pvalue,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        human_prob=human_prob,
        nonhuman_prob=nonhuman_prob,
        score=score,
    )

    conclusion = {"response": int(score), "explanation": explanation}

    # Write a single JSON object with no extra lines.
    with open("conclusion.txt", "w") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()
