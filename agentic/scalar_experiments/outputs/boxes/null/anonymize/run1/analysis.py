import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def likelihood_ratio_test(ll_full: float, ll_reduced: float, df_diff: int):
    """Return LR test statistic and p-value comparing two nested models."""
    lr_stat = 2.0 * (ll_full - ll_reduced)
    p_value = stats.chi2.sf(lr_stat, df=df_diff)
    return lr_stat, p_value


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Derived variables
    df["majority_choice"] = (df["choice"] == 2).astype(int)
    df["social_choice"] = df["choice"].isin([2, 3]).astype(int)

    # Basic descriptives
    descriptives = {}
    descriptives["n"] = int(len(df))
    descriptives["overall_majority_rate"] = float(df["majority_choice"].mean())
    descriptives["overall_social_rate"] = float(df["social_choice"].mean())

    # Majority choice ~ age + site
    logit_majority_full = smf.logit("majority_choice ~ age + C(site)", data=df).fit(
        disp=False
    )
    logit_majority_no_age = smf.logit("majority_choice ~ C(site)", data=df).fit(
        disp=False
    )
    logit_majority_no_site = smf.logit("majority_choice ~ age", data=df).fit(
        disp=False
    )

    # LR tests
    n_sites = df["site"].nunique()
    lr_age_majority, p_age_majority = likelihood_ratio_test(
        logit_majority_full.llf, logit_majority_no_age.llf, df_diff=1
    )
    lr_site_majority, p_site_majority = likelihood_ratio_test(
        logit_majority_full.llf, logit_majority_no_site.llf, df_diff=n_sites - 1
    )

    # Social choice (any demonstrated option) ~ age + site
    logit_social_full = smf.logit("social_choice ~ age + C(site)", data=df).fit(
        disp=False
    )
    logit_social_no_age = smf.logit("social_choice ~ C(site)", data=df).fit(disp=False)
    logit_social_no_site = smf.logit("social_choice ~ age", data=df).fit(disp=False)

    lr_age_social, p_age_social = likelihood_ratio_test(
        logit_social_full.llf, logit_social_no_age.llf, df_diff=1
    )
    lr_site_social, p_site_social = likelihood_ratio_test(
        logit_social_full.llf, logit_social_no_site.llf, df_diff=n_sites - 1
    )

    # Age effect directions (slopes)
    age_coef_majority = float(logit_majority_full.params["age"])
    age_coef_social = float(logit_social_full.params["age"])

    # Majority choice by age (simple correlation for effect size)
    r_age_majority, p_corr_age_majority = stats.pearsonr(df["age"], df["majority_choice"])
    r_age_social, p_corr_age_social = stats.pearsonr(df["age"], df["social_choice"])

    # Majority choice by site (proportions)
    majority_by_site = (
        df.groupby("site")["majority_choice"].mean().sort_index().to_dict()
    )
    social_by_site = df.groupby("site")["social_choice"].mean().sort_index().to_dict()

    results = {
        "descriptives": descriptives,
        "majority_model": {
            "lr_age_stat": lr_age_majority,
            "lr_age_p": p_age_majority,
            "lr_site_stat": lr_site_majority,
            "lr_site_p": p_site_majority,
            "age_coef": age_coef_majority,
            "r_age_majority": float(r_age_majority),
            "r_age_majority_p": float(p_corr_age_majority),
            "majority_rate_by_site": majority_by_site,
        },
        "social_model": {
            "lr_age_stat": lr_age_social,
            "lr_age_p": p_age_social,
            "lr_site_stat": lr_site_social,
            "lr_site_p": p_site_social,
            "age_coef": age_coef_social,
            "r_age_social": float(r_age_social),
            "r_age_social_p": float(p_corr_age_social),
            "social_rate_by_site": social_by_site,
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

