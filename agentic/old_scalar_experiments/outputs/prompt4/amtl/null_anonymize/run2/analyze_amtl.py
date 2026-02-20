import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def infer_missing_total(df: pd.DataFrame) -> tuple[str, str]:
    """
    Infer which column is 'missing' and which is 'total' by checking
    which assignment yields no (or fewer) cases where missing > total.
    """
    col3, col4 = "feature3", "feature4"

    invalid_3_as_missing = (df[col3] > df[col4]).sum()
    invalid_4_as_missing = (df[col4] > df[col3]).sum()

    # Prefer the orientation with fewer invalid rows; if tied, default to
    # treating feature4 as missing, feature3 as total (matches first rows).
    if invalid_3_as_missing < invalid_4_as_missing:
        missing_col, total_col = col3, col4
    else:
        missing_col, total_col = col4, col3

    return missing_col, total_col


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_col, total_col = infer_missing_total(df)

    df = df.copy()
    df["missing"] = df[missing_col].astype(float)
    df["total"] = df[total_col].astype(float)

    # Keep only rows with a sensible denominator
    df = df[df["total"] > 0].copy()
    df = df[df["missing"] >= 0].copy()
    df = df[df["missing"] <= df["total"]].copy()

    df["prop_missing"] = df["missing"] / df["total"]

    # Covariates
    df["is_human"] = df["feature8"].astype(str).str.startswith("Homo").astype(int)
    df["age"] = df["feature5"].astype(float)
    df["sex_est"] = df["feature7"].astype(float)
    df["tooth_class"] = df["feature1"].astype("category")

    return df


def fit_glm(df: pd.DataFrame):
    # Binomial GLM with logit link; use proportions with frequency weights
    formula = "prop_missing ~ is_human + age + sex_est + C(tooth_class)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total"],
    )
    result = model.fit()
    return result


def summarize_by_genus(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("feature8")
        .agg(
            n_specimens=("feature2", "nunique"),
            n_rows=("feature2", "size"),
            total_teeth=("total", "sum"),
            total_missing=("missing", "sum"),
        )
        .reset_index()
    )
    summary["prop_missing"] = summary["total_missing"] / summary["total_teeth"]
    return summary


def compute_support_score(coef: float, p_value: float) -> int:
    """
    Map the effect size and significance for 'is_human' to a 0-100 Likert score.
    Positive coefficient -> humans have higher AMTL.
    """
    # Basic direction:
    if coef > 0:
        base = 75
    else:
        base = 25

    # Adjust for magnitude (on log-odds scale)
    magnitude = min(abs(coef), 2.5) / 2.5  # cap to avoid extremes

    # Adjust for p-value (smaller p => stronger evidence)
    if p_value < 0.001:
        sig_factor = 1.0
    elif p_value < 0.01:
        sig_factor = 0.8
    elif p_value < 0.05:
        sig_factor = 0.6
    elif p_value < 0.1:
        sig_factor = 0.4
    else:
        sig_factor = 0.2

    if coef > 0:
        score = base + 25 * magnitude * sig_factor
    else:
        score = base - 25 * magnitude * sig_factor

    # Ensure integer within [0, 100]
    return int(round(max(0, min(100, score))))


def main():
    data_path = Path("amtl.csv")
    df = load_and_prepare(data_path)

    genus_summary = summarize_by_genus(df)
    glm_result = fit_glm(df)

    coef_human = glm_result.params.get("is_human", np.nan)
    p_human = glm_result.pvalues.get("is_human", np.nan)

    # Build explanation text
    lines = []
    lines.append(
        "I analyzed antemortem tooth loss (AMTL) using a binomial regression "
        "model on counts of missing teeth versus observable sockets for each specimen and tooth class."
    )
    lines.append(
        "The model used a logit link and included predictors for whether the specimen was Homo sapiens "
        "(versus Pan, Papio, or Pongo), estimated age at death, estimated sex, and tooth class "
        "(anterior, posterior, premolar)."
    )

    # Add genus-level descriptive stats
    for _, row in genus_summary.iterrows():
        genus = row["feature8"]
        prop = row["prop_missing"]
        lines.append(
            f"For genus {genus}, the overall proportion of teeth missing was approximately {prop:.3f}."
        )

    if np.isfinite(coef_human) and np.isfinite(p_human):
        odds_ratio = float(np.exp(coef_human))
        direction = (
            "higher" if coef_human > 0 else "lower" if coef_human < 0 else "similar"
        )
        lines.append(
            "In the regression model, the coefficient for the Homo sapiens indicator "
            f"was {coef_human:.3f} on the log-odds scale (odds ratio ≈ {odds_ratio:.2f}, "
            f"p-value = {p_human:.4f})."
        )
        lines.append(
            f"This means that, after adjusting for age, sex, and tooth class, modern humans show {direction} "
            "frequencies of antemortem tooth loss compared to the non-human primates in this dataset."
        )
        support_score = compute_support_score(coef_human, p_human)
    else:
        lines.append(
            "The model could not estimate a reliable effect for Homo sapiens; "
            "there is insufficient evidence in this dataset to distinguish human AMTL frequencies "
            "from those of non-human primates after adjustment."
        )
        support_score = 50

    explanation = " ".join(lines)

    conclusion = {
        "response": int(support_score),
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

