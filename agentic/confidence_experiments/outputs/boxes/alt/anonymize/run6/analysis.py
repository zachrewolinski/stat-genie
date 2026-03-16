import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def likelihood_ratio_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    return float(lr_stat), float(df_diff), float(p_value)


def summarize_probabilities(model, df, outcome_col):
    """
    Compute predicted probabilities over age and site for high-level description.
    """
    ages = np.linspace(df["age"].min(), df["age"].max(), 5)
    sites = sorted(df["site"].unique())
    summary = {}
    for site in sites:
        site_preds = []
        for age in ages:
            row = {"age": age, "site": site}
            pred_prob = model.predict(pd.DataFrame([row]))[0]
            site_preds.append(pred_prob)
        summary[int(site)] = {
            "min_age": float(ages.min()),
            "max_age": float(ages.max()),
            "prob_min_age": float(site_preds[0]),
            "prob_max_age": float(site_preds[-1]),
        }
    overall_probs = model.predict(
        pd.DataFrame(
            {
                "age": df["age"],
                "site": df["site"],
            }
        )
    )
    overall = {
        "mean_prob": float(overall_probs.mean()),
        "min_prob": float(overall_probs.min()),
        "max_prob": float(overall_probs.max()),
    }
    return {"by_site": summary, "overall": overall}


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

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
    df["site"] = df["site"].astype(int)

    # Derived variables
    df["social_choice"] = df["outcome"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["outcome"] == 2).astype(int)

    # Social vs asocial model
    social_full = smf.glm(
        "social_choice ~ age + C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    social_no_age = smf.glm(
        "social_choice ~ C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    social_no_site = smf.glm(
        "social_choice ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    social_lr_age = likelihood_ratio_test(social_full, social_no_age)
    social_lr_site = likelihood_ratio_test(social_full, social_no_site)
    social_probs = summarize_probabilities(
        social_full, df[["age", "site", "social_choice"]].copy(), "social_choice"
    )

    # Majority vs other model
    majority_full = smf.glm(
        "majority_choice ~ age + C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    majority_no_age = smf.glm(
        "majority_choice ~ C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    majority_no_site = smf.glm(
        "majority_choice ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    majority_lr_age = likelihood_ratio_test(majority_full, majority_no_age)
    majority_lr_site = likelihood_ratio_test(majority_full, majority_no_site)
    majority_probs = summarize_probabilities(
        majority_full, df[["age", "site", "majority_choice"]].copy(), "majority_choice"
    )

    results = {
        "n": int(len(df)),
        "age_range": [float(df["age"].min()), float(df["age"].max())],
        "num_sites": int(df["site"].nunique()),
        "social_lr_age": {
            "lr_stat": social_lr_age[0],
            "df": social_lr_age[1],
            "p_value": social_lr_age[2],
        },
        "social_lr_site": {
            "lr_stat": social_lr_site[0],
            "df": social_lr_site[1],
            "p_value": social_lr_site[2],
        },
        "majority_lr_age": {
            "lr_stat": majority_lr_age[0],
            "df": majority_lr_age[1],
            "p_value": majority_lr_age[2],
        },
        "majority_lr_site": {
            "lr_stat": majority_lr_site[0],
            "df": majority_lr_site[1],
            "p_value": majority_lr_site[2],
        },
        "social_probs": social_probs,
        "majority_probs": majority_probs,
    }

    # Save numerical results to inspect from the outside
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also print a compact summary
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

