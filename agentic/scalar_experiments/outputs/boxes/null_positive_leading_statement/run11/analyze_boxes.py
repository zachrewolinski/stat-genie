import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Compute Cramer's V from chi-square."""
    if n == 0:
        return 0.0
    k = min(r - 1, c - 1)
    if k <= 0:
        return 0.0
    return math.sqrt(chi2 / (n * k))


def chi2_majority_vs_factor(df: pd.DataFrame, factor: str):
    """Chi-square test of majority choice vs a categorical factor."""
    contingency = pd.crosstab(df["majority_choice"], df[factor])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    v = cramers_v(chi2, contingency.to_numpy().sum(), *contingency.shape)
    return chi2, p, v


def classify_evidence(p_age: float, v_age: float, p_cult: float, v_cult: float) -> int:
    """Map statistical evidence to a Likert score in [-100, 100]."""
    # Count dimensions with conventional significance
    sig_001 = sum(p < 0.001 for p in (p_age, p_cult))
    sig_01 = sum(0.001 <= p < 0.01 for p in (p_age, p_cult))
    sig_05 = sum(0.01 <= p < 0.05 for p in (p_age, p_cult))

    strong_effects = sum(v >= 0.25 for v in (v_age, v_cult))
    moderate_effects = sum(0.15 <= v < 0.25 for v in (v_age, v_cult))

    # Strong "yes": consistent, robust evidence across dimensions
    if sig_001 >= 1 and (strong_effects >= 1 or (sig_001 + sig_01) >= 2):
        return 85

    # Moderate "yes": clear but somewhat smaller or uneven effects
    if (sig_01 + sig_001) >= 1 and (strong_effects + moderate_effects) >= 1:
        return 60

    # Weak "yes": at least some conventional significance
    if (sig_05 + sig_01 + sig_001) >= 1:
        return 30

    # No clear evidence either way
    return 0


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Outcome recode: majority vs non-majority
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Age groups approximating developmental stages
    bins = [4, 6, 8, 10, 12, 14]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-13"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)

    # Drop any rare missing bins to avoid empty columns in crosstabs
    df = df.dropna(subset=["age_group"])

    # Chi-square tests
    chi2_age, p_age, v_age = chi2_majority_vs_factor(df, "age_group")
    chi2_cult, p_cult, v_cult = chi2_majority_vs_factor(df, "culture")

    # Derive Likert scalar
    score = classify_evidence(p_age, v_age, p_cult, v_cult)

    # For transparency during development, print core stats
    print("N =", len(df))
    print("Majority-choice rate overall:", df["majority_choice"].mean())
    print("Chi2 age_group:", chi2_age, "p_age:", p_age, "V_age:", v_age)
    print("Chi2 culture:", chi2_cult, "p_cult:", p_cult, "V_cult:", v_cult)
    print("Derived Likert score:", score)

    # Write conclusion.txt with ONLY the integer scalar value
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(int(score)), encoding="utf-8")


if __name__ == "__main__":
    main()

