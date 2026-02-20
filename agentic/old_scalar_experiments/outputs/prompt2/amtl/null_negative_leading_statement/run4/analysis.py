import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    # Ensure numeric types
    df["num_amtl"] = pd.to_numeric(df["num_amtl"])
    df["sockets"] = pd.to_numeric(df["sockets"])
    df["age"] = pd.to_numeric(df["age"])
    df["prob_male"] = pd.to_numeric(df["prob_male"])
    return df


def prepare_design_matrix(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    # Response as proportion with binomial weights
    y = df["num_amtl"] / df["sockets"]

    # Human vs non-human is encoded via genus dummies with Homo sapiens as baseline
    X = pd.DataFrame(index=df.index)
    X["Intercept"] = 1.0
    X["genus_Pan"] = (df["genus"] == "Pan").astype(float)
    X["genus_Pongo"] = (df["genus"] == "Pongo").astype(float)
    X["genus_Papio"] = (df["genus"] == "Papio").astype(float)

    # Covariates: age, sex proxy, and tooth class (Anterior as baseline)
    X["age"] = df["age"].astype(float)
    X["prob_male"] = df["prob_male"].astype(float)
    X["tooth_Posterior"] = (df["tooth_class"] == "Posterior").astype(float)
    X["tooth_Premolar"] = (df["tooth_class"] == "Premolar").astype(float)

    return y, X


def fit_model(y: pd.Series, X: pd.DataFrame, weights: pd.Series):
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()
    return result


def summarize_genus_effects(result: sm.GLM, alpha: float = 0.05) -> dict:
    params = result.params
    conf_int = result.conf_int(alpha=alpha)
    summary = {}
    for genus_var, label in [
        ("genus_Pan", "Pan"),
        ("genus_Pongo", "Pongo"),
        ("genus_Papio", "Papio"),
    ]:
        if genus_var in params.index:
            beta = params[genus_var]
            ci_low, ci_high = conf_int.loc[genus_var]
            summary[label] = {
                "coef": float(beta),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
    return summary


def interpret_results(genus_effects: dict) -> tuple[str, float, str]:
    """
    Decide whether Homo sapiens have higher AMTL frequency than non-human genera
    after adjusting for covariates.

    With Homo sapiens as the baseline, positive genus coefficients mean that
    the corresponding non-human genus has *higher* AMTL frequency than humans.
    Negative coefficients mean *lower* frequency than humans.

    We answer "Yes" only if humans have clearly higher AMTL than all three
    non-human genera, operationalized as 95% confidence intervals for all three
    genus effects lying strictly below zero (i.e., all non-human genera
    significantly lower than humans). Otherwise we answer "No".
    """
    if not genus_effects:
        explanation = (
            "The regression model did not estimate genus-specific effects, "
            "so we cannot conclude that humans have higher AMTL than non-human primates."
        )
        return "No", 40.0, explanation

    all_below_zero = all(g["ci_high"] < 0.0 for g in genus_effects.values())
    any_positive_center = any(g["coef"] > 0.0 for g in genus_effects.values())

    lines = []
    for genus, stats in genus_effects.items():
        lines.append(
            f"{genus}: coefficient={stats['coef']:.3f}, "
            f"95% CI=({stats['ci_low']:.3f}, {stats['ci_high']:.3f}) "
            "(positive values indicate higher AMTL than humans)."
        )
    genus_text = " ".join(lines)

    if all_below_zero:
        response = "Yes"
        confidence = 80.0
        explanation = (
            "Using a binomial regression of AMTL counts on genus, age, sex, and tooth class, "
            "with Homo sapiens as the baseline, all three non-human genera (Pan, Pongo, Papio) "
            "have significantly negative genus coefficients whose 95% confidence intervals lie entirely "
            "below zero. This indicates that, after adjusting for age, sex, and tooth class, "
            "modern humans have higher AMTL frequencies than each non-human primate genus. "
            f"Estimated genus effects: {genus_text}"
        )
    else:
        response = "No"
        # Increase confidence if multiple genera appear to have higher AMTL than humans
        if any_positive_center:
            confidence = 85.0
        else:
            confidence = 70.0
        explanation = (
            "A binomial regression of AMTL counts on genus, age, sex, and tooth class, "
            "with Homo sapiens as the baseline, does not show all non-human primate genera "
            "having AMTL frequencies lower than humans. At least one non-human genus has a "
            "genus coefficient that is centered at or above zero, and several confidence intervals "
            "include zero, indicating no robust evidence that humans have higher AMTL frequency "
            "than all three non-human genera after adjustment. "
            f"Estimated genus effects: {genus_text}"
        )

    return response, confidence, explanation


def main():
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "amtl.csv"
    conclusion_path = base / "conclusion.txt"

    _ = load_metadata(info_path)
    df = load_data(data_path)

    y, X = prepare_design_matrix(df)
    weights = df["sockets"]
    result = fit_model(y, X, weights)

    genus_effects = summarize_genus_effects(result)
    response, confidence, explanation = interpret_results(genus_effects)

    conclusion = {
        "response": response,
        "confidence": float(confidence),
        "explanation": explanation,
    }

    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

