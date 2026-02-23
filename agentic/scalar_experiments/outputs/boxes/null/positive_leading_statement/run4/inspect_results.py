import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np


def cramers_v(chi2: float, n: int, min_dim: int) -> float:
    if n <= 0 or min_dim <= 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * min_dim)))


def main() -> None:
    df = pd.read_csv("boxes.csv")
    df["social"] = df["y"].isin([2, 3]).astype(int)
    social_df = df[df["social"] == 1].copy()
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

    bins = [3, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_stage"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
    social_df["age_stage"] = pd.cut(social_df["age"], bins=bins, labels=labels, include_lowest=True)

    print("N:", df.shape[0])

    ct_culture_social = pd.crosstab(df["culture"], df["social"])
    chi2_cs, p_cs, dof_cs, exp_cs = chi2_contingency(ct_culture_social)
    v_cs = cramers_v(chi2_cs, df.shape[0], min(ct_culture_social.shape) - 1)
    print("culture_social chi2, p, V:", chi2_cs, p_cs, v_cs)

    ct_stage_social = pd.crosstab(df["age_stage"], df["social"])
    chi2_as, p_as, dof_as, exp_as = chi2_contingency(ct_stage_social)
    v_as = cramers_v(chi2_as, df.shape[0], min(ct_stage_social.shape) - 1)
    print("age_social chi2, p, V:", chi2_as, p_as, v_as)

    ct_culture_majority = pd.crosstab(social_df["culture"], social_df["majority_choice"])
    chi2_cm, p_cm, dof_cm, exp_cm = chi2_contingency(ct_culture_majority)
    v_cm = cramers_v(chi2_cm, social_df.shape[0], min(ct_culture_majority.shape) - 1)
    print("culture_majority chi2, p, V:", chi2_cm, p_cm, v_cm)

    ct_stage_majority = pd.crosstab(social_df["age_stage"], social_df["majority_choice"])
    chi2_am, p_am, dof_am, exp_am = chi2_contingency(ct_stage_majority)
    v_am = cramers_v(chi2_am, social_df.shape[0], min(ct_stage_majority.shape) - 1)
    print("age_majority chi2, p, V:", chi2_am, p_am, v_am)


if __name__ == "__main__":
    main()

