import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_observed",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic sanity filters
    df = df[df["n_observed"] > 0].copy()
    df = df[df["n_missing"] >= 0].copy()
    df = df[df["n_missing"] <= df["n_observed"]].copy()

    # Indicator for modern humans
    df["is_human"] = df["genus"].astype(str).str.contains("Homo", case=False, na=False)
    df["is_human"] = df["is_human"].astype(int)

    return df


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each specimen/tooth-class row into per-tooth rows so that a
    standard logistic regression can be fit on a binary AMTL outcome.
    """
    records = []

    for _, row in df.iterrows():
        n_missing = int(row["n_missing"])
        n_observed = int(row["n_observed"])

        if n_observed <= 0 or n_missing < 0 or n_missing > n_observed:
            continue

        base = {
            "tooth_class": row["tooth_class"],
            "age": float(row["age"]),
            "age_uncertainty": float(row["age_uncertainty"]),
            "sex_estimate": float(row["sex_estimate"]),
            "is_human": int(row["is_human"]),
        }

        for _ in range(n_missing):
            rec = base.copy()
            rec["amtl"] = 1
            records.append(rec)

        for _ in range(n_observed - n_missing):
            rec = base.copy()
            rec["amtl"] = 0
            records.append(rec)

    return pd.DataFrame.from_records(records)


def fit_model(df_long: pd.DataFrame):
    """
    Fit a binomial (logistic) regression for AMTL at the tooth level.
    """
    formula = "amtl ~ is_human + C(tooth_class) + age + sex_estimate"
    model = smf.glm(formula=formula, data=df_long, family=sm.families.Binomial())
    result = model.fit()
    return result


def assess_human_effect(model_result) -> dict:
    """
    Extract the effect of being human vs non-human and compute
    a binary answer and confidence score.
    """
    params = model_result.params
    pvalues = model_result.pvalues
    conf_int = model_result.conf_int()

    if "is_human" not in params:
        response = "No"
        confidence = 40
        explanation = (
            "The regression model could not estimate a separate effect for humans "
            "versus non-human primates (no 'is_human' coefficient was present), "
            "so there is insufficient evidence in this dataset to conclude that "
            "humans differ in AMTL frequency after accounting for age, sex, and tooth class."
        )
        return {
            "response": response,
            "confidence": confidence,
            "explanation": explanation,
        }

    coef = float(params["is_human"])
    pval = float(pvalues["is_human"])
    ci_low, ci_high = conf_int.loc["is_human"]

    odds_ratio = float(np.exp(coef))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    if coef > 0 and pval < 0.05:
        response = "Yes"
    else:
        response = "No"

    if pval < 0.001:
        confidence = 95
    elif pval < 0.01:
        confidence = 90
    elif pval < 0.05:
        confidence = 80
    elif pval < 0.1:
        confidence = 65
    else:
        confidence = 50

    explanation_lines = []

    explanation_lines.append(
        "I modeled the probability that an individual tooth was lost antemortem "
        "(AMTL = 1 vs 0) using a binomial (logistic) regression fitted to the "
        "expanded tooth-level dataset constructed from the provided counts of "
        "missing teeth and observable sockets."
    )
    explanation_lines.append(
        "The model included predictors for whether the specimen was a modern human "
        "(Homo; indicator is_human), tooth class (anterior, posterior, premolar), "
        "age at death (continuous), and estimated sex, thereby accounting for age, "
        "sex, and tooth class while estimating the human effect."
    )
    explanation_lines.append(
        f"The estimated coefficient for the human indicator (is_human) was {coef:.3f}, "
        f"corresponding to an odds ratio of {odds_ratio:.2f} for AMTL in humans relative "
        f"to non-human primates, with a 95% confidence interval for the odds ratio of "
        f"[{or_low:.2f}, {or_high:.2f}] and a p-value of {pval:.3g}."
    )

    if response == "Yes":
        explanation_lines.append(
            "Because the human indicator has a positive coefficient and is statistically "
            "significant (p < 0.05), this indicates that, after controlling for age, "
            "sex, and tooth class, modern humans in this sample exhibit higher "
            "frequencies of antemortem tooth loss than the combined non-human primate "
            "genera (Pan, Pongo, Papio)."
        )
    else:
        if coef <= 0:
            explanation_lines.append(
                "The estimated human effect is non-positive, indicating that, after "
                "controlling for age, sex, and tooth class, humans do not show higher "
                "AMTL frequencies than non-human primates in this dataset."
            )
        else:
            explanation_lines.append(
                "Although the estimated human effect is positive, the p-value is not "
                "below the conventional 0.05 threshold, so the data do not provide "
                "sufficient evidence that humans have higher AMTL frequencies than "
                "non-human primates once age, sex, and tooth class are accounted for."
            )

    explanation = " ".join(explanation_lines)

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def write_conclusion(path: Path, result: dict) -> None:
    obj = {
        "response": result["response"],
        "confidence": int(result["confidence"]),
        "explanation": str(result["explanation"]),
    }
    text = json.dumps(obj, ensure_ascii=False)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    base_dir = Path(".")
    info_path = base_dir / "info.json"
    data_path = base_dir / "amtl.csv"
    conclusion_path = base_dir / "conclusion.txt"

    _ = load_metadata(info_path)
    df = load_data(data_path)
    df_long = expand_to_tooth_level(df)

    if df_long.empty:
        result = {
            "response": "No",
            "confidence": 30,
            "explanation": (
                "After attempting to construct a tooth-level dataset from the provided "
                "AMTL counts, no valid observations remained (e.g., due to inconsistent "
                "or missing counts), so it is not possible to determine whether humans "
                "have higher AMTL frequencies than non-human primates from this data."
            ),
        }
    else:
        model_result = fit_model(df_long)
        result = assess_human_effect(model_result)

    write_conclusion(conclusion_path, result)


if __name__ == "__main__":
    main()

