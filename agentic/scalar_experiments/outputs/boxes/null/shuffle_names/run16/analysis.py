import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Recode outcomes
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = (df["majority_first"] != 1).astype(int)

    # Treat site id `y` as categorical culture indicator
    df["site"] = df["y"].astype("category")

    results = {}

    # Majority preference: logistic regression with age and site
    majority_model = smf.logit(
        formula="majority_choice ~ age + C(site)", data=df
    ).fit(disp=False)
    results["majority_age_coef"] = float(majority_model.params["age"])
    results["majority_age_p"] = float(majority_model.pvalues["age"])
    results["majority_llr_p"] = float(majority_model.llr_pvalue)

    # Likelihood ratio test for site effects: compare model with and without site
    majority_model_reduced = smf.logit(
        formula="majority_choice ~ age", data=df
    ).fit(disp=False)
    lr_stat = 2 * (majority_model.llf - majority_model_reduced.llf)
    df_diff = majority_model.df_model - majority_model_reduced.df_model
    lr_p = stats.chi2.sf(lr_stat, df_diff)
    results["majority_site_lr_p"] = float(lr_p)

    # Social information reliance: logistic regression with age and site
    social_model = smf.logit(
        formula="social_choice ~ age + C(site)", data=df
    ).fit(disp=False)
    results["social_age_coef"] = float(social_model.params["age"])
    results["social_age_p"] = float(social_model.pvalues["age"])
    results["social_llr_p"] = float(social_model.llr_pvalue)

    social_model_reduced = smf.logit(
        formula="social_choice ~ age", data=df
    ).fit(disp=False)
    lr_stat_social = 2 * (social_model.llf - social_model_reduced.llf)
    df_diff_social = social_model.df_model - social_model_reduced.df_model
    lr_p_social = stats.chi2.sf(lr_stat_social, df_diff_social)
    results["social_site_lr_p"] = float(lr_p_social)

    # Descriptive statistics by age and site for context
    by_site = (
        df.groupby("site")[["majority_choice", "social_choice"]]
        .mean()
        .rename(
            columns={
                "majority_choice": "majority_rate",
                "social_choice": "social_rate",
            }
        )
    )
    results["by_site"] = by_site.to_dict(orient="index")

    # Simple correlation between age and outcomes
    results["age_majority_corr"], results["age_majority_corr_p"] = map(
        float, stats.pearsonr(df["age"], df["majority_choice"])
    )
    results["age_social_corr"], results["age_social_corr_p"] = map(
        float, stats.pearsonr(df["age"], df["social_choice"])
    )

    # Save intermediate stats for manual inspection if needed
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
