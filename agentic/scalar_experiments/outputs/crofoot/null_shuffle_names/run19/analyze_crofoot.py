import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct interpretable predictors based on the metadata descriptions:
    - group_size_diff: focal group size minus other group size
    - location_advantage: positive when focal group is closer to its home-range center
    """
    df = df.copy()

    # From info.json descriptions:
    # f_other: number of individuals in focal group
    # win:    number of individuals in other group
    df["group_size_focal"] = df["f_other"]
    df["group_size_other"] = df["win"]
    df["group_size_diff"] = df["group_size_focal"] - df["group_size_other"]

    # m_other: distance of focal group from center of its home range
    # n_focal: distance of other group from center of its home range
    df["dist_focal_center"] = df["m_other"]
    df["dist_other_center"] = df["n_focal"]
    # Positive when focal is closer (other is further from its center)
    df["location_advantage"] = df["dist_other_center"] - df["dist_focal_center"]

    return df


def standardize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[col + "_z"] = 0.0
        else:
            df[col + "_z"] = (df[col] - mean) / std
    return df


def p_to_weight(p: float) -> float:
    """
    Map p-values into [0, 1] weights reflecting strength of evidence.
    """
    if p < 0.001:
        return 1.0
    if p < 0.01:
        return 0.8
    if p < 0.05:
        return 0.6
    if p < 0.1:
        return 0.4
    if p < 0.2:
        return 0.2
    return 0.0


def compute_likert_scalar(df: pd.DataFrame) -> int:
    """
    Fit a logistic regression of contest outcome (focal win)
    on relative group size and location advantage, then map
    the combined evidence to a Likert scalar in [-100, 100].
    """
    df = build_features(df)
    predictors = ["group_size_diff", "location_advantage"]
    df = standardize_columns(df, predictors)

    y = df["m_focal"]
    X = df[[col + "_z" for col in predictors]]
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X).fit(disp=False)

    coefs = model.params
    pvalues = model.pvalues

    # McFadden pseudo-R^2 as an overall effect-size summary
    llf = model.llf
    llnull = model.llnull
    pseudo_r2 = max(0.0, 1.0 - llf / llnull) if llnull != 0 else 0.0

    # Evidence from each predictor: direction-consistent, p-value-weighted
    component_scores: list[float] = []
    for base_name in predictors:
        col = base_name + "_z"
        coef = float(coefs[col])
        p = float(pvalues[col])
        weight = p_to_weight(p)

        # Positive coefficient means: larger group-size difference or stronger
        # location advantage raises win probability, which answers the research
        # question in the "Yes, they matter" direction.
        if coef > 0:
            component_scores.append(weight)
        elif coef < 0:
            component_scores.append(-weight)
        else:
            component_scores.append(0.0)

    # Average over predictors for a symmetric [-1, 1] core score
    if component_scores:
        predictor_score = float(np.mean(component_scores))
    else:
        predictor_score = 0.0

    # Overall model fit contributes up to +0.4 towards "Yes"
    r2_contrib = min(pseudo_r2, 0.4)

    combined_score = predictor_score + r2_contrib
    combined_score = max(-1.0, min(1.0, combined_score))

    scalar = int(round(combined_score * 100))
    scalar = max(-100, min(100, scalar))
    return scalar


def main() -> None:
    data_path = Path("crofoot.csv")
    df = load_data(data_path)

    scalar = compute_likert_scalar(df)

    # Write the scalar conclusion as the only contents of conclusion.txt
    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

