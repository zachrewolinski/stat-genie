import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)
    return df


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data).fit(disp=False)
    return model


def summarize_effect_strength(model, var_prefix: str):
    """
    Compute a simple summary of evidence that a predictor (or group of dummies)
    matters in the model. We use the proportion of coefficients with p<0.05 and
    the median absolute z-value for those coefficients whose names start with
    `var_prefix`.
    """
    params = model.params
    pvalues = model.pvalues
    zvalues = model.tvalues  # for Logit, tvalues are z-stats

    mask = [name.startswith(var_prefix) or name == var_prefix for name in params.index]
    if not any(mask):
        return 0.0

    p_sel = pvalues[mask]
    z_sel = zvalues[mask].abs()

    if len(p_sel) == 0:
        return 0.0

    sig_mask = p_sel < 0.05
    if not sig_mask.any():
        return 0.0

    prop_sig = sig_mask.mean()
    median_z = float(z_sel[sig_mask].median())

    # Simple combined score: more significant coefficients and larger z -> stronger evidence
    score = prop_sig * min(median_z / 3.0, 1.0)  # cap z contribution
    return score


def main():
    df = load_data()

    # Construct key outcome variables
    df["social_choice"] = (df["y"] != 1).astype(int)  # chose majority or minority vs undemonstrated
    df["majority_choice"] = (df["y"] == 2).astype(int)  # majority vs all others

    # Majority vs minority only among children who used social information
    df_social = df[df["y"].isin([2, 3])].copy()
    df_social["majority_over_minority"] = (df_social["y"] == 2).astype(int)

    evidence_scores = []

    # Model 1: reliance on social information
    try:
        m_social = fit_logit("social_choice ~ age + C(culture) + C(gender) + majority_first", df)
        age_score_social = summarize_effect_strength(m_social, "age")
        culture_score_social = summarize_effect_strength(m_social, "C(culture)")
        evidence_scores.extend([age_score_social, culture_score_social])
    except Exception:
        pass

    # Model 2: majority vs others
    try:
        m_majority = fit_logit("majority_choice ~ age + C(culture) + C(gender) + majority_first", df)
        age_score_majority = summarize_effect_strength(m_majority, "age")
        culture_score_majority = summarize_effect_strength(m_majority, "C(culture)")
        evidence_scores.extend([age_score_majority, culture_score_majority])
    except Exception:
        pass

    # Model 3: majority vs minority among social choosers
    try:
        if len(df_social["majority_over_minority"].unique()) > 1:
            m_pref = fit_logit(
                "majority_over_minority ~ age + C(culture) + C(gender) + majority_first",
                df_social,
            )
            age_score_pref = summarize_effect_strength(m_pref, "age")
            culture_score_pref = summarize_effect_strength(m_pref, "C(culture)")
            evidence_scores.extend([age_score_pref, culture_score_pref])
    except Exception:
        pass

    # Aggregate evidence that age and/or culture matter
    if evidence_scores:
        avg_score = float(np.mean(evidence_scores))
    else:
        avg_score = 0.0

    # Map evidence score in [0,1] roughly to Likert scale.
    # Strong evidence -> strong "Yes" (variation across age/culture).
    scalar = int(round(80 * avg_score))  # cap typical values around +/-80

    # We are assessing: "Do children’s reliance and majority preference vary?"
    # If evidence is present, that supports "Yes"; otherwise near 0.
    scalar = max(min(scalar, 100), -100)

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(f"{scalar}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

