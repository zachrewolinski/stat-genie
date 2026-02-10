import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def load_data(csv_path: str = "boxes.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Outcome coding: 1 = undemonstrated, 2 = majority, 3 = minority
    df = df.copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["minority_choice"] = (df["y"] == 3).astype(int)
    df["unchosen_choice"] = (df["y"] == 1).astype(int)
    return df


def analyse_majority_by_age_culture(df: pd.DataFrame):
    # Basic sanity: drop any missing values in key predictors
    df_model = df[["majority_choice", "age", "culture"]].dropna()

    # Ensure numeric age and categorical culture
    df_model["age"] = pd.to_numeric(df_model["age"], errors="coerce")
    df_model = df_model.dropna(subset=["age"])

    # Fit reduced model: majority_choice ~ age
    m_age = smf.logit("majority_choice ~ age", data=df_model).fit(disp=False)

    # Fit full model: majority_choice ~ age + C(culture)
    m_full = smf.logit("majority_choice ~ age + C(culture)", data=df_model).fit(disp=False)

    # Age effect: p-value from full model and effect size across age range
    age_p = float(m_full.pvalues.get("age", np.nan))
    age_vals = df_model["age"]
    age_low, age_high = np.percentile(age_vals, [20, 80])
    age_grid = df_model.copy()
    age_grid["age"] = age_low
    p_low = m_full.predict(age_grid).mean()
    age_grid["age"] = age_high
    p_high = m_full.predict(age_grid).mean()
    age_effect = float(p_high - p_low)

    # Culture effect: likelihood-ratio test comparing models with and without culture
    lr_stat = 2 * (m_full.llf - m_age.llf)
    df_diff = m_full.df_model - m_age.df_model
    culture_p = float(stats.chi2.sf(lr_stat, df_diff))

    # Culture effect size: variation in mean majority choice by culture
    culture_means = (
        df_model.groupby("culture")["majority_choice"]
        .mean()
        .sort_index()
    )
    culture_range = float(culture_means.max() - culture_means.min())

    return {
        "age_p": age_p,
        "age_effect": age_effect,
        "culture_p": culture_p,
        "culture_range": culture_range,
    }


def map_to_scalar(evidence: dict) -> int:
    age_p = evidence["age_p"]
    age_effect = abs(evidence["age_effect"])
    culture_p = evidence["culture_p"]
    culture_range = evidence["culture_range"]

    score = 0.0

    # Age-related variation in majority choice
    if not math.isnan(age_p):
        if age_p < 0.05 and age_effect > 0.05:
            score += 25
        if age_p < 0.01 and age_effect > 0.10:
            score += 15

    # Culture-related variation in majority choice
    if not math.isnan(culture_p):
        if culture_p < 0.05 and culture_range > 0.10:
            score += 35
        if culture_p < 0.01 and culture_range > 0.20:
            score += 25

    # If we saw essentially no evidence of variation, lean toward a mild "No"
    if score == 0.0:
        score = -20.0

    # Clamp to [-100, 100] and round to integer
    score = int(round(max(-100.0, min(100.0, score))))
    return score


def main():
    df = load_data()
    evidence = analyse_majority_by_age_culture(df)
    scalar = map_to_scalar(evidence)

    # Write scalar conclusion to file, as required
    out_path = Path("conclusion.txt")
    out_path.write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

