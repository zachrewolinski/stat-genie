import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    base_dir = Path(__file__).parent

    info_path = base_dir / "info.json"
    data_path = base_dir / "boxes.csv"

    info = load_metadata(info_path)
    df = pd.read_csv(data_path)

    # Outcome: 1 = undemonstrated, 2 = majority, 3 = minority
    df = df.copy()
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Basic sanity: drop rows with missing key fields if any
    df = df.dropna(subset=["majority_choice", "age", "y"])

    # ----- Age effect: logistic regression -----
    # Standardize age to improve numerical stability
    age = df["age"].astype(float)
    age_z = (age - age.mean()) / age.std(ddof=0)
    y_majority = df["majority_choice"]

    X_age = sm.add_constant(age_z)
    try:
        model_age = sm.Logit(y_majority, X_age).fit(disp=False)
        age_p = model_age.pvalues.get("age", np.nan)
        age_beta = model_age.params.get("age", np.nan)
    except Exception:
        age_p = np.nan
        age_beta = np.nan

    # Effect size via difference in majority rates between youngest and oldest terciles
    df["age_tercile"] = pd.qcut(df["age"], q=3, labels=[0, 1, 2])
    age_group_rates = (
        df.groupby("age_tercile")["majority_choice"].mean().sort_index()
    )
    if len(age_group_rates) == 3:
        age_rate_diff = float(age_group_rates.iloc[2] - age_group_rates.iloc[0])
    else:
        age_rate_diff = 0.0

    # ----- Cultural (site) effect: chi-square across sites -----
    site_col = "y"  # site ID from 1 to 8
    contingency = pd.crosstab(df[site_col], df["majority_choice"])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_site, _, _ = stats.chi2_contingency(contingency)
    else:
        p_site = np.nan
        chi2 = np.nan

    site_rates = df.groupby(site_col)["majority_choice"].mean()
    if len(site_rates) > 1:
        site_rate_range = float(site_rates.max() - site_rates.min())
    else:
        site_rate_range = 0.0

    # ----- Demonstration order effect (majority demonstrated first or not) -----
    # This captures reliance on presented social information structure.
    order_col = "culture"  # 0/1 according to metadata description
    if order_col in df.columns:
        contingency_order = pd.crosstab(df[order_col], df["majority_choice"])
        if contingency_order.shape[0] > 1 and contingency_order.shape[1] > 1:
            chi2_order, p_order, _, _ = stats.chi2_contingency(contingency_order)
        else:
            p_order = np.nan
            chi2_order = np.nan
        order_rates = df.groupby(order_col)["majority_choice"].mean()
        if len(order_rates) > 1:
            order_rate_diff = float(order_rates.max() - order_rates.min())
        else:
            order_rate_diff = 0.0
    else:
        p_order = np.nan
        chi2_order = np.nan
        order_rate_diff = 0.0

    # ----- Map results to Likert scalar [-100, 100] -----
    # We interpret strong, consistent variation across age and cultures
    # as strong evidence for the research question.

    score = 0.0

    # Age contribution
    if not np.isnan(age_p):
        if age_p < 0.001 and abs(age_rate_diff) >= 0.20:
            score += 40
        elif age_p < 0.01 and abs(age_rate_diff) >= 0.15:
            score += 30
        elif age_p < 0.05 and abs(age_rate_diff) >= 0.10:
            score += 20
        elif age_p >= 0.1 or abs(age_rate_diff) < 0.05:
            # Weak or absent age differences
            score -= 10

    # Cultural/site contribution
    if not np.isnan(p_site):
        if p_site < 0.001 and site_rate_range >= 0.25:
            score += 40
        elif p_site < 0.01 and site_rate_range >= 0.20:
            score += 30
        elif p_site < 0.05 and site_rate_range >= 0.15:
            score += 20
        elif p_site >= 0.1 or site_rate_range < 0.05:
            score -= 10

    # Demonstration order contribution – supplementary to main question
    if not np.isnan(p_order):
        if p_order < 0.01 and order_rate_diff >= 0.15:
            score += 10
        elif p_order >= 0.1 or order_rate_diff < 0.05:
            score -= 5

    # Clamp to [-100, 100] and round to nearest integer
    score = int(max(-100, min(100, round(score))))

    conclusion_path = base_dir / "conclusion.txt"
    conclusion_path.write_text(str(score), encoding="utf-8")


if __name__ == "__main__":
    main()

