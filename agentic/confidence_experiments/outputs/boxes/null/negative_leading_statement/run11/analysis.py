import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Define derived variables
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    mask_demonstrated = df["y"].isin([2, 3])
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))
    df["demonstrated"] = mask_demonstrated.astype(int)
    return df


def fit_glm(formula: str, data: pd.DataFrame):
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
    result = model.fit()
    return result


def lr_test(full_model, reduced_model):
    lr_stat = 2.0 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    if df_diff <= 0:
        p_value = np.nan
    else:
        p_value = chi2.sf(lr_stat, df_diff)
    return float(lr_stat), float(p_value)


def summarize_effect_ranges(df: pd.DataFrame):
    # Simple descriptive ranges to contextualize any detected effects
    by_culture = (
        df.groupby("culture")
        .agg(
            social_rate=("social_choice", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )

    # Age groups in 3-year bins
    bins = [4, 7, 10, 13, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    by_age = (
        df.groupby("age_group")
        .agg(
            social_rate=("social_choice", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )

    return by_culture, by_age


def main():
    csv_path = Path("boxes.csv")
    df = load_data(csv_path)

    # Model 1: reliance on social information (choosing any demonstrated option)
    m1_full = fit_glm("social_choice ~ age + C(culture)", df)
    m1_no_age = fit_glm("social_choice ~ C(culture)", df)
    m1_no_culture = fit_glm("social_choice ~ age", df)

    m1_age_lr, m1_age_p = lr_test(m1_full, m1_no_age)
    m1_culture_lr, m1_culture_p = lr_test(m1_full, m1_no_culture)

    # Model 2: majority preference among children who followed a demonstration
    df_demonstrated = df[df["demonstrated"] == 1].copy()
    m2_full = fit_glm("majority_choice ~ age + C(culture)", df_demonstrated)
    m2_no_age = fit_glm("majority_choice ~ C(culture)", df_demonstrated)
    m2_no_culture = fit_glm("majority_choice ~ age", df_demonstrated)

    m2_age_lr, m2_age_p = lr_test(m2_full, m2_no_age)
    m2_culture_lr, m2_culture_p = lr_test(m2_full, m2_no_culture)

    by_culture, by_age = summarize_effect_ranges(df)

    summary = {
        "n": int(len(df)),
        "n_demonstrated": int(df["demonstrated"].sum()),
        "model1_social_choice": {
            "age_lr_stat": m1_age_lr,
            "age_lr_pvalue": m1_age_p,
            "culture_lr_stat": m1_culture_lr,
            "culture_lr_pvalue": m1_culture_p,
        },
        "model2_majority_choice": {
            "age_lr_stat": m2_age_lr,
            "age_lr_pvalue": m2_age_p,
            "culture_lr_stat": m2_culture_lr,
            "culture_lr_pvalue": m2_culture_p,
        },
        "by_culture": by_culture.to_dict(orient="list"),
        "by_age_group": by_age.to_dict(orient="list"),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
