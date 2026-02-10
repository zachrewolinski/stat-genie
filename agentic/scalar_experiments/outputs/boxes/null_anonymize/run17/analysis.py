import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


DATA_FILE = "boxes.csv"
OUTPUT_FILE = "conclusion.txt"


def p_to_score(p: float) -> float:
    """Map a p-value to an evidence score in [-1, 1]."""
    if math.isnan(p):
        return 0.0
    if p < 1e-6:
        return 1.0
    if p < 1e-3:
        return 0.8
    if p < 1e-2:
        return 0.6
    if p < 5e-2:
        return 0.4
    if p < 1e-1:
        return 0.2
    # If p is very large, treat as weak evidence against variation
    if p > 0.5:
        return -0.2
    return 0.0


def lr_test(full_model, reduced_model) -> float:
    """Likelihood-ratio test p-value between nested models."""
    try:
        lr_stat, lr_p, _ = full_model.compare_lr_test(reduced_model)
        return float(lr_p)
    except Exception:
        return float("nan")


def chi_square_p(table: pd.DataFrame) -> float:
    """Fallback chi-square test p-value for contingency tables."""
    try:
        _, p, _, _ = stats.chi2_contingency(table)
        return float(p)
    except Exception:
        return float("nan")


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "outcome",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Derived variables
    df["social_use"] = df["outcome"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["outcome"] == 2).astype(int)
    df["site"] = df["site"].astype("category")
    df["gender"] = df["gender"].astype("category")
    df["majority_first"] = df["majority_first"].astype(int)

    # 1) Reliance on social information ~ age + site
    p_social_age = float("nan")
    p_social_site = float("nan")
    try:
        m_social_full = smf.logit(
            "social_use ~ age + C(site) + majority_first", data=df
        ).fit(disp=False)
        m_social_no_age = smf.logit(
            "social_use ~ C(site) + majority_first", data=df
        ).fit(disp=False)
        m_social_no_site = smf.logit(
            "social_use ~ age + majority_first", data=df
        ).fit(disp=False)

        p_social_age = lr_test(m_social_full, m_social_no_age)
        p_social_site = lr_test(m_social_full, m_social_no_site)
    except Exception:
        # Fallback: chi-square tests
        df["age_bin"] = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=False)
        table_age = pd.crosstab(df["age_bin"], df["social_use"])
        table_site = pd.crosstab(df["site"], df["social_use"])
        p_social_age = chi_square_p(table_age)
        p_social_site = chi_square_p(table_site)

    # 2) Majority preference among social users ~ age + site
    social = df[df["social_use"] == 1].copy()
    p_majority_age = float("nan")
    p_majority_site = float("nan")
    if len(social) > 0 and social["majority_choice"].nunique() > 1:
        try:
            m_maj_full = smf.logit(
                "majority_choice ~ age + C(site) + majority_first", data=social
            ).fit(disp=False)
            m_maj_no_age = smf.logit(
                "majority_choice ~ C(site) + majority_first", data=social
            ).fit(disp=False)
            m_maj_no_site = smf.logit(
                "majority_choice ~ age + majority_first", data=social
            ).fit(disp=False)

            p_majority_age = lr_test(m_maj_full, m_maj_no_age)
            p_majority_site = lr_test(m_maj_full, m_maj_no_site)
        except Exception:
            df["age_bin"] = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=False)
            table_age = pd.crosstab(social["age_bin"], social["majority_choice"])
            table_site = pd.crosstab(social["site"], social["majority_choice"])
            p_majority_age = chi_square_p(table_age)
            p_majority_site = chi_square_p(table_site)

    # Map p-values to evidence scores
    scores = []
    for p in [p_social_age, p_social_site, p_majority_age, p_majority_site]:
        scores.append(p_to_score(p))

    # Average evidence across all four aspects
    mean_score = float(np.nanmean(scores))
    mean_score = max(-1.0, min(1.0, mean_score))
    scalar = int(round(mean_score * 100))

    # Ensure we produce a scalar in [-100, 100]
    scalar = max(-100, min(100, scalar))

    Path(OUTPUT_FILE).write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

