import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


def load_data():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)
    return df


def compute_evidence_scalar(df: pd.DataFrame) -> int:
    """
    Compute an integer scalar in [-100, 100] indicating support for:
    "Children's reliance on social information and preference for majority cues
    vary across cultures and developmental stages."
    """
    df = df.copy()
    # Outcome: 1=unchosen, 2=majority, 3=minority
    df["choose_majority"] = (df["majority_first"] == 2).astype(int)

    # --- Age-related variation (developmental stages) ---
    q1 = df["age"].quantile(0.25)
    q3 = df["age"].quantile(0.75)
    young = df[df["age"] <= q1]
    old = df[df["age"] >= q3]

    young_rate = young["choose_majority"].mean()
    old_rate = old["choose_majority"].mean()
    age_diff = float(old_rate - young_rate)

    # Logistic regression for age effect
    age_model = smf.logit("choose_majority ~ age", data=df).fit(disp=False)
    p_age = float(age_model.pvalues.get("age", 1.0))

    # Scale age effect size to [0, 1], capping at a 0.3 difference
    age_effect_strength = max(age_diff, 0.0) / 0.3
    age_effect_strength = float(np.clip(age_effect_strength, 0.0, 1.0))

    if p_age < 0.01:
        age_sig_weight = 1.0
    elif p_age < 0.05:
        age_sig_weight = 0.7
    else:
        age_sig_weight = 0.4

    age_score = age_effect_strength * age_sig_weight

    # --- Cultural variation (sites as cultural contexts) ---
    site_means = df.groupby("y")["choose_majority"].mean()
    site_diff = float(site_means.max() - site_means.min())

    # Chi-square test of independence between site and majority choice
    contingency = pd.crosstab(df["y"], df["choose_majority"])
    _, p_site, _, _ = chi2_contingency(contingency)
    p_site = float(p_site)

    site_effect_strength = max(site_diff, 0.0) / 0.3
    site_effect_strength = float(np.clip(site_effect_strength, 0.0, 1.0))

    if p_site < 0.01:
        site_sig_weight = 1.0
    elif p_site < 0.05:
        site_sig_weight = 0.7
    else:
        site_sig_weight = 0.4

    site_score = site_effect_strength * site_sig_weight

    # Combine age and cultural components into overall evidence e in [0, 1]
    combined_score = age_score + site_score
    max_possible = age_sig_weight + site_sig_weight
    if max_possible > 0:
        e = combined_score / max_possible
    else:
        e = 0.5  # fallback to neutral if something goes wrong

    e = float(np.clip(e, 0.0, 1.0))

    # Map e \in [0,1] to scalar s \in [-100,100]
    scalar = int(round((e - 0.5) * 200))

    # Ensure bounds
    scalar = max(min(scalar, 100), -100)

    # Print some basic diagnostics for transparency (not written to conclusion.txt)
    print("N =", len(df))
    print(f"Young majority rate: {young_rate:.3f}, Old majority rate: {old_rate:.3f}")
    print(f"Age diff: {age_diff:.3f}, p_age: {p_age:.3g}, age_score: {age_score:.3f}")
    print(
        f"Site majority rate range: {site_means.min():.3f}–{site_means.max():.3f}, "
        f"diff: {site_diff:.3f}, p_site: {p_site:.3g}, site_score: {site_score:.3f}"
    )
    print(f"Combined evidence e: {e:.3f}, scalar: {scalar}")

    return scalar


def main():
    df = load_data()
    scalar = compute_evidence_scalar(df)

    # Write the final scalar to conclusion.txt with no extra text.
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(scalar) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

