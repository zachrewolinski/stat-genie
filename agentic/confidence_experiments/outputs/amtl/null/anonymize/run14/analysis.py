import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: str = "info.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_data(path: str = "amtl.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "observable",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_score",
            "feature8": "genus",
            "feature9": "region",
        }
    )
    # Keep only rows with valid main variables and positive number of observable sockets
    required = ["missing", "observable", "age", "sex_score", "tooth_class", "genus"]
    df = df.dropna(subset=required)
    df = df[df["observable"] > 0].copy()

    # Restrict to genera of interest
    valid_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(valid_genera)].copy()

    # Proportion of missing teeth within a tooth class for each specimen
    df["prop_missing"] = df["missing"] / df["observable"]
    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial regression: missing teeth / observable sockets
    ~ is_human + age + sex_score + tooth_class.
    """
    if df.empty:
        return None

    # Use GLM with binomial family. Model the proportion with frequency weights.
    try:
        model = smf.glm(
            formula="prop_missing ~ is_human + age + sex_score + C(tooth_class)",
            data=df,
            family=sm.families.Binomial(),
            freq_weights=df["observable"],
        ).fit()
    except Exception:
        # Fallback without frequency weights if something goes wrong
        model = smf.glm(
            formula="prop_missing ~ is_human + age + sex_score + C(tooth_class)",
            data=df,
            family=sm.families.Binomial(),
        ).fit()
    return model


def summarize_effect(
    df: pd.DataFrame, model
) -> Tuple[float, float, float, float, float]:
    """
    Return coefficient, p-value for is_human and predicted probabilities
    for representative humans vs non-human primates.
    """
    import math

    if model is None:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    coef = model.params.get("is_human", np.nan)
    pval = model.pvalues.get("is_human", np.nan)

    tooth_counts = df["tooth_class"].value_counts(normalize=True)
    mean_age = df["age"].mean()
    mean_sex = df["sex_score"].mean()

    def avg_pred(is_human_value: int) -> float:
        rows = []
        for tc, _w in tooth_counts.items():
            rows.append(
                {
                    "is_human": is_human_value,
                    "age": mean_age,
                    "sex_score": mean_sex,
                    "tooth_class": tc,
                }
            )
        new_df = pd.DataFrame(rows)
        preds = model.predict(new_df)
        return float(np.average(preds, weights=tooth_counts.values))

    try:
        human_prob = avg_pred(1)
        nonhuman_prob = avg_pred(0)
        diff_prob = human_prob - nonhuman_prob
    except Exception:
        human_prob = nonhuman_prob = diff_prob = np.nan

    return coef, pval, human_prob, nonhuman_prob, diff_prob


def map_to_likert(coef: float, pval: float, diff_prob: float) -> Tuple[int, str, bool]:
    """
    Map evidence about the human effect to a 0-100 Likert score
    and a textual Yes/No label.
    """
    import math

    if not (math.isfinite(coef) and math.isfinite(pval)):
        return 50, "inconclusive", False

    # Base strength from p-value (smaller p -> stronger evidence)
    if pval < 0.001:
        strength = 25
    elif pval < 0.01:
        strength = 20
    elif pval < 0.05:
        strength = 15
    elif pval < 0.1:
        strength = 10
    else:
        strength = 5

    # Effect-size contribution from difference in predicted probabilities
    if math.isfinite(diff_prob):
        effect_mag = min(abs(diff_prob), 0.3) / 0.3  # cap at 0.3 absolute difference
    else:
        effect_mag = min(abs(coef), 2.0) / 2.0

    strength = strength + int(20 * effect_mag)

    if pval < 0.05 and coef > 0:
        response = min(100, 70 + strength)
        yn = "Yes"
        evidence_for_higher = True
    elif pval < 0.05 and coef <= 0:
        response = max(0, 30 - strength)
        yn = "No"
        evidence_for_higher = False
    elif pval >= 0.05 and coef > 0:
        # Directionally higher but not statistically significant
        response = max(0, min(100, 50 + strength))
        yn = "No"
        evidence_for_higher = False
    else:
        # Directionally lower or essentially no pattern and not significant
        response = max(0, min(100, 50 - strength))
        yn = "No"
        evidence_for_higher = False

    return int(response), yn, evidence_for_higher


def build_explanation(
    research_q: str,
    df: pd.DataFrame,
    coef: float,
    pval: float,
    human_prob: float,
    nonhuman_prob: float,
    diff_prob: float,
    yn: str,
    evidence_for_higher: bool,
) -> str:
    parts = []
    parts.append(f"Research question: {research_q}")
    parts.append(
        "I analyzed the AMTL dataset (missing teeth and observable sockets per specimen and tooth class) "
        "using binomial regression to test whether modern humans have higher frequencies of antemortem "
        "tooth loss than non-human primates while adjusting for age at death, sex estimate, and tooth class."
    )
    parts.append(
        f"The analysis included {len(df)} tooth-class observations across four genera "
        "(Homo sapiens, Pan, Papio, Pongo). The response was the proportion of missing teeth "
        "per tooth class (missing / observable sockets), modeled with a binomial GLM with logit link."
    )

    if np.isfinite(coef) and np.isfinite(pval):
        parts.append(
            f"The estimated coefficient for the modern human indicator on the log-odds scale was {coef:.3f} "
            f"with p-value {pval:.3g} after controlling for age, sex score, and tooth class."
        )
    else:
        parts.append(
            "The model did not return a well-defined coefficient or p-value for the modern human indicator, "
            "so evidence for a difference is treated as inconclusive."
        )

    if np.isfinite(human_prob) and np.isfinite(nonhuman_prob):
        parts.append(
            "Based on the fitted model, at average age and sex and averaging over tooth classes, the "
            f"estimated probability of AMTL was {human_prob:.3f} for modern humans and "
            f"{nonhuman_prob:.3f} for non-human primates (difference {diff_prob:.3f} in absolute probability)."
        )

    if evidence_for_higher:
        summary = (
            "These results provide statistically significant evidence that modern humans have higher AMTL "
            "frequencies than the non-human primate genera studied, even after accounting for age, sex, and "
            "tooth class."
        )
    else:
        summary = (
            "Taken together, these results do not provide convincing evidence that modern humans have higher "
            "AMTL frequencies than non-human primates once age, sex, and tooth class are accounted for in the model."
        )
    parts.append(summary)

    parts.append(
        f"Overall conclusion: {yn} — modern humans "
        f"{'do' if evidence_for_higher else 'do not'} show clearly higher AMTL frequencies than non-human primates "
        "in this dataset under the binomial regression model used."
    )

    return " ".join(parts)


def write_output(response: int, explanation: str, path: str = "conclusion.txt") -> None:
    obj = {"response": int(response), "explanation": explanation}
    with open(path, "w") as f:
        json.dump(obj, f)


def main() -> None:
    meta = load_metadata()
    research_q = meta.get("research_questions", [""])[0]

    df = load_data()
    model = fit_model(df)

    coef, pval, human_prob, nonhuman_prob, diff_prob = summarize_effect(df, model)
    response, yn, evidence_for_higher = map_to_likert(coef, pval, diff_prob)
    explanation = build_explanation(
        research_q,
        df,
        coef,
        pval,
        human_prob,
        nonhuman_prob,
        diff_prob,
        yn,
        evidence_for_higher,
    )
    write_output(response, explanation)


if __name__ == "__main__":
    main()

