import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata():
    """Load research question and basic metadata from info.json."""
    info_path = Path("info.json")
    with info_path.open() as f:
        info = json.load(f)

    questions = info.get("research_questions", [])
    research_question = questions[0] if questions else ""
    return research_question


def load_data():
    """Load AMTL dataset from CSV."""
    df = pd.read_csv("amtl.csv")

    # Drop rows with missing key variables
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )

    # Keep rows with valid socket counts and counts within [0, sockets]
    df = df[
        (df["sockets"] > 0)
        & (df["num_amtl"] >= 0)
        & (df["num_amtl"] <= df["sockets"])
    ].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Center age and sex-probability for more interpretable coefficients
    df["age_c"] = df["age"] - df["age"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    return df


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the design matrix for the GLM with:
    - intercept
    - is_human
    - age_c
    - prob_male_c
    - tooth_class (dummy-coded, reference = first level)
    """
    X = pd.DataFrame(
        {
            "const": 1.0,
            "is_human": df["is_human"].astype(float),
            "age_c": df["age_c"].astype(float),
            "prob_male_c": df["prob_male_c"].astype(float),
        }
    )
    tooth_dummies = pd.get_dummies(
        df["tooth_class"], prefix="tooth_class", drop_first=True
    )
    X = pd.concat([X, tooth_dummies], axis=1)
    return X


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial regression model for AMTL counts with predictors:
    - is_human (modern human vs non-human primate)
    - age
    - sex (prob_male)
    - tooth class

    The response is specified as a two-column matrix [successes, failures].
    """
    # Successes and failures per row
    successes = df["num_amtl"].astype(float)
    failures = (df["sockets"] - df["num_amtl"]).astype(float)
    endog = np.column_stack([successes, failures])

    X = build_design_matrix(df)

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()
    return result


def summarize_genus_rates(df: pd.DataFrame):
    """Compute observed AMTL proportions by genus (num_amtl / sockets)."""
    grouped = df.groupby("genus", observed=True)
    summary = []
    for genus, g in grouped:
        total_missing = g["num_amtl"].sum()
        total_sockets = g["sockets"].sum()
        prop = float(total_missing / total_sockets) if total_sockets > 0 else np.nan
        summary.append(
            {
                "genus": genus,
                "n_rows": int(len(g)),
                "total_missing": int(total_missing),
                "total_sockets": int(total_sockets),
                "prop_missing": prop,
            }
        )
    return summary


def adjusted_predictions(result, df: pd.DataFrame):
    """
    Compute adjusted predicted AMTL probabilities for humans vs non-humans
    by setting is_human to 1 or 0 for all observations while holding
    age, sex, and tooth class at their observed values.
    """
    base_df = df.copy()
    human_df = base_df.copy()
    human_df["is_human"] = 1
    nonhuman_df = base_df.copy()
    nonhuman_df["is_human"] = 0

    X_human = build_design_matrix(human_df)
    X_nonhuman = build_design_matrix(nonhuman_df)

    pred_human = float(result.predict(X_human).mean())
    pred_nonhuman = float(result.predict(X_nonhuman).mean())
    return pred_human, pred_nonhuman


def compute_likert_score(coef_is_human: float, p_is_human: float, or_is_human: float):
    """
    Map evidence about the human effect to a 0–100 Likert score where:
    - 0   = very strong evidence that humans do NOT have higher AMTL
    - 50  = indeterminate / equivocal
    - 100 = very strong evidence that humans DO have higher AMTL
    """
    # No clear evidence either direction
    if not np.isfinite(or_is_human) or p_is_human >= 0.05:
        return 40  # mild "No" / inconclusive

    # Significant effect: direction matters
    if coef_is_human > 0:
        # Humans have higher AMTL
        # Clip odds ratio to a reasonable range for scaling
        or_clipped = float(np.clip(or_is_human, 1.0, 4.0))
        effect = (or_clipped - 1.0) / (4.0 - 1.0)  # 0–1
        score = 70 + effect * 30  # 70–100
    else:
        # Humans have significantly lower AMTL than non-humans
        # Work with inverse OR to describe how much lower
        or_inv = float(np.clip(1.0 / or_is_human, 1.0, 4.0))
        effect = (or_inv - 1.0) / (4.0 - 1.0)  # 0–1
        score = 10 + (1.0 - effect) * 30  # 10–40, stronger evidence -> closer to 10

    score_int = int(round(score))
    return max(0, min(100, score_int))


def build_explanation(
    research_question: str,
    df: pd.DataFrame,
    genus_summary,
    result,
    pred_human: float,
    pred_nonhuman: float,
    coef_is_human: float,
    p_is_human: float,
    or_is_human: float,
):
    n_total = int(len(df))
    n_human = int(df["is_human"].sum())
    n_nonhuman = n_total - n_human

    explanation_lines = []
    explanation_lines.append(
        "Research question: "
        + research_question
        + " The dataset contains per-specimen counts of missing teeth "
        "and observable sockets, along with age, sex (as probability of male), "
        "tooth class, and primate genus."
    )
    explanation_lines.append(
        f"Sample size is {n_total} tooth-class observations: "
        f"{n_human} for modern humans (Homo sapiens) and {n_nonhuman} for non-human primate genera."
    )

    # Observed genus-level proportions
    explanation_lines.append(
        "Observed (unadjusted) AMTL proportions by genus (missing teeth / sockets):"
    )
    for s in genus_summary:
        genus = s["genus"]
        prop = s["prop_missing"]
        explanation_lines.append(
            f"  - {genus}: {prop:.3f} proportion of missing teeth "
            f"({s['total_missing']} missing out of {s['total_sockets']} sockets; {s['n_rows']} rows)."
        )

    # Model-based adjusted results
    explanation_lines.append(
        "To control for age, sex, and tooth class, I fit a binomial regression "
        "model predicting the proportion of missing teeth with predictors: "
        "an indicator for modern humans vs non-human primates, centered age, "
        "centered probability of being male, and categorical tooth class."
    )
    explanation_lines.append(
        "In this model, the coefficient for the human indicator represents "
        "the log-odds difference in AMTL between humans and non-human primates, "
        "after adjusting for age, sex, and tooth class."
    )

    explanation_lines.append(
        f"The estimated log-odds coefficient for modern humans is {coef_is_human:.3f}, "
        f"which corresponds to an odds ratio of {or_is_human:.3f} "
        f"with p-value {p_is_human:.4g}."
    )
    explanation_lines.append(
        "I also computed adjusted predicted AMTL probabilities by setting all "
        "observations to human or non-human while keeping age, sex, and tooth "
        "class at their observed values."
    )
    explanation_lines.append(
        f"Under this adjustment, the mean predicted probability of a tooth being missing "
        f"is {pred_human:.3f} for humans and {pred_nonhuman:.3f} for non-human primates."
    )

    if p_is_human < 0.05 and coef_is_human > 0:
        qualitative = (
            "These results provide statistically significant evidence that humans have higher "
            "frequencies of antemortem tooth loss than the non-human primate genera considered, "
            "even after adjusting for age, sex, and tooth class."
        )
    elif p_is_human < 0.05 and coef_is_human <= 0:
        qualitative = (
            "These results provide statistically significant evidence that humans do not have "
            "higher frequencies of antemortem tooth loss than non-human primates; if anything, "
            "their AMTL frequencies are lower after adjusting for age, sex, and tooth class."
        )
    else:
        qualitative = (
            "The human indicator is not statistically significant at conventional levels "
            "after adjusting for age, sex, and tooth class, so this dataset does not provide "
            "strong evidence that humans differ from non-human primates in AMTL frequency."
        )
    explanation_lines.append(qualitative)

    return " ".join(explanation_lines)


def main():
    research_question = load_metadata()
    df = load_data()

    # Descriptive summaries
    genus_summary = summarize_genus_rates(df)

    # Model fitting
    result = fit_model(df)

    # Human effect
    coef_is_human = float(result.params["is_human"])
    p_is_human = float(result.pvalues["is_human"])
    or_is_human = float(np.exp(coef_is_human))

    # Adjusted predictions
    pred_human, pred_nonhuman = adjusted_predictions(result, df)

    # Likert response score
    response_score = compute_likert_score(coef_is_human, p_is_human, or_is_human)

    # Explanation text
    explanation = build_explanation(
        research_question=research_question,
        df=df,
        genus_summary=genus_summary,
        result=result,
        pred_human=pred_human,
        pred_nonhuman=pred_nonhuman,
        coef_is_human=coef_is_human,
        p_is_human=p_is_human,
        or_is_human=or_is_human,
    )

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    # Write JSON output to conclusion.txt (single JSON object, no extra text)
    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
