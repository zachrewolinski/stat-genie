import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata (for transparency / potential future use)
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    # Core dataset
    data_path = base_dir / "boxes.csv"
    df = pd.read_csv(data_path)

    # feature1: 1=undemonstrated, 2=majority option, 3=minority option
    majority = (df["feature1"] == 2).astype(int)
    age = df["feature3"].astype(float)
    site = df["feature5"].astype(int)

    # --- Developmental variation: majority preference vs age ---
    # Use point-biserial correlation (equivalent to Pearson with binary variable)
    try:
        r_age, p_age = stats.pointbiserialr(majority, age)
    except Exception:
        r_age, p_age = 0.0, 1.0

    r_age = float(np.nan_to_num(r_age, nan=0.0))
    p_age = float(np.nan_to_num(p_age, nan=1.0))

    # Effect-size scaling: treat |r|=0.3 as "moderate" (strength=1.0 cap)
    age_strength = min(abs(r_age) / 0.3, 1.0)

    # --- Cross-cultural variation: majority preference vs site (culture proxy) ---
    site_majority_rate = df.groupby(site)["feature1"].apply(
        lambda x: (x == 2).mean()
    )
    max_rate = float(site_majority_rate.max())
    min_rate = float(site_majority_rate.min())
    diff_site = max_rate - min_rate

    # Effect-size scaling: 0.3 difference in majority rate ~ "moderate"
    site_strength = min(diff_site / 0.3, 1.0)

    # Chi-square test of independence between site and majority choice
    contingency = pd.crosstab(site, majority)
    try:
        chi2, p_site, _, _ = stats.chi2_contingency(contingency)
    except Exception:
        chi2, p_site = 0.0, 1.0

    p_site = float(np.nan_to_num(p_site, nan=1.0))

    # --- Combine evidence into a Likert-style scalar ---
    # Base strength from standardized effect sizes
    base_strength = (age_strength + site_strength) / 2.0

    # P-value modulation: reward stronger statistical evidence
    def p_weight(p: float) -> float:
        if p < 0.001:
            return 1.0
        if p < 0.05:
            return 0.7
        if p < 0.2:
            return 0.4
        return 0.2

    p_factor_age = p_weight(p_age)
    p_factor_site = p_weight(p_site)
    p_factor = (p_factor_age + p_factor_site) / 2.0

    strength = base_strength * p_factor  # in [0, 1] approximately

    # Decide whether the answer leans toward "Yes, they vary" or "No"
    evidence_for_variation = (
        (p_age < 0.05 and abs(r_age) > 0.1)
        or (p_site < 0.05 and diff_site > 0.1)
    )

    if evidence_for_variation:
        sign = 1
    else:
        # If effects are tiny and non-significant, lean toward "No" or neutral
        if base_strength < 0.15 and p_age > 0.2 and p_site > 0.2:
            # Clear lack of evidence: moderately negative
            sign = -1
            # So that extremely tiny base_strength does not collapse to zero
            strength = max(strength, 0.3)
        else:
            # Ambiguous/mixed evidence: neutral-ish
            sign = 0

    # Map strength in [0,1] to [0,100], then apply sign
    scalar = int(round(strength * 100)) * sign

    # Clip to [-100, 100] to obey Likert bounds
    scalar = max(-100, min(100, scalar))

    # Write final scalar to conclusion.txt with no extra text
    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

