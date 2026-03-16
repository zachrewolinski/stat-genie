import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


@dataclass
class ModelEffectSummary:
    p_age: float | None
    p_site: float | None
    age_effect_direction: str | None
    site_range: float | None
    overall_rate: float


def likelihood_ratio_test(full_model, reduced_model, df_diff: int) -> float:
    """Return p-value for a likelihood ratio test comparing two nested models."""
    llr = 2 * (full_model.llf - reduced_model.llf)
    return float(chi2.sf(llr, df_diff))


def summarize_binary_outcome(df: pd.DataFrame, outcome_col: str) -> float:
    """Compute overall mean for a binary outcome column."""
    return float(df[outcome_col].mean())


def analyze_social_use(df: pd.DataFrame) -> ModelEffectSummary:
    """Logistic regression for using social information at all."""
    df = df.copy()
    df["social_use"] = (df["choice"] != 1).astype(int)

    formula_full = "social_use ~ age + C(site) + C(gender) + majority_first"
    model_full = smf.logit(formula_full, data=df).fit(disp=False)

    # Reduced models for LR tests
    model_no_age = smf.logit(
        "social_use ~ C(site) + C(gender) + majority_first", data=df
    ).fit(disp=False)
    model_no_site = smf.logit(
        "social_use ~ age + C(gender) + majority_first", data=df
    ).fit(disp=False)

    p_age = float(model_full.pvalues.get("age", np.nan))
    p_site = likelihood_ratio_test(
        model_full, model_no_site, df_diff=model_full.df_model - model_no_site.df_model
    )

    # Direction of age effect
    age_coef = model_full.params.get("age", np.nan)
    if np.isnan(age_coef):
        age_dir = None
    elif age_coef > 0:
        age_dir = "increasing"
    else:
        age_dir = "decreasing"

    # Range of site-level means
    site_means = df.groupby("site")["social_use"].mean()
    site_range = float(site_means.max() - site_means.min())

    overall_rate = summarize_binary_outcome(df, "social_use")

    return ModelEffectSummary(
        p_age=p_age,
        p_site=p_site,
        age_effect_direction=age_dir,
        site_range=site_range,
        overall_rate=overall_rate,
    )


def analyze_majority_preference(df: pd.DataFrame) -> ModelEffectSummary:
    """Logistic regression for preferring majority over minority when using social information."""
    df = df.copy()
    social_mask = df["choice"].isin([2, 3])
    df_social = df.loc[social_mask].copy()

    df_social["majority_choice"] = (df_social["choice"] == 2).astype(int)

    formula_full = "majority_choice ~ age + C(site) + C(gender) + majority_first"
    model_full = smf.logit(formula_full, data=df_social).fit(disp=False)

    model_no_site = smf.logit(
        "majority_choice ~ age + C(gender) + majority_first", data=df_social
    ).fit(disp=False)

    p_age = float(model_full.pvalues.get("age", np.nan))
    p_site = likelihood_ratio_test(
        model_full, model_no_site, df_diff=model_full.df_model - model_no_site.df_model
    )

    age_coef = model_full.params.get("age", np.nan)
    if np.isnan(age_coef):
        age_dir = None
    elif age_coef > 0:
        age_dir = "increasing"
    else:
        age_dir = "decreasing"

    site_means = df_social.groupby("site")["majority_choice"].mean()
    site_range = float(site_means.max() - site_means.min())

    overall_rate = summarize_binary_outcome(df_social, "majority_choice")

    return ModelEffectSummary(
        p_age=p_age,
        p_site=p_site,
        age_effect_direction=age_dir,
        site_range=site_range,
        overall_rate=overall_rate,
    )


def main() -> None:
    df = pd.read_csv("boxes.csv")
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )
    df["gender"] = df["gender"].map({1: "girl", 2: "boy"})
    df["site"] = df["site"].astype("category")

    social_summary = analyze_social_use(df)
    majority_summary = analyze_majority_preference(df)

    # Descriptive age-binned summaries
    df["social_use"] = (df["choice"] != 1).astype(int)
    age_bins = [4, 7, 10, 13, 15]  # [4-6], [7-9], [10-12], [13-14]
    age_labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, right=False, labels=age_labels)
    social_by_age = {
        str(k): float(v) for k, v in df.groupby("age_group")["social_use"].mean().items()
    }

    social_mask = df["choice"].isin([2, 3])
    df_social = df.loc[social_mask].copy()
    df_social["majority_choice"] = (df_social["choice"] == 2).astype(int)
    df_social["age_group"] = pd.cut(
        df_social["age"], bins=age_bins, right=False, labels=age_labels
    )
    majority_by_age = {
        str(k): float(v)
        for k, v in df_social.groupby("age_group")["majority_choice"].mean().items()
    }

    output = {
        "social_use": asdict(social_summary),
        "majority_preference": asdict(majority_summary),
        "descriptives": {
            "social_use_by_age_group": social_by_age,
            "majority_pref_by_age_group": majority_by_age,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
