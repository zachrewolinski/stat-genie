import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Outcome coding:
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["social_info"] = (df["majority_first"] != 1).astype(int)
    df_social = df.copy()

    # Majority preference among children who relied on social information
    df_majority = df[df["social_info"] == 1].copy()
    df_majority["majority_choice"] = (df_majority["majority_first"] == 2).astype(
        int
    )

    # Treat site ID as categorical proxy for cultural context
    df["site"] = df["y"].astype("category")
    df_social["site"] = df_social["y"].astype("category")
    df_majority["site"] = df_majority["y"].astype("category")

    # Descriptive summaries
    overall_social_rate = df_social["social_info"].mean()
    overall_majority_rate = df_majority["majority_choice"].mean()

    social_by_site = df_social.groupby("site")["social_info"].mean()
    majority_by_site = df_majority.groupby("site")["majority_choice"].mean()

    # Age as continuous predictor
    # Model 1: reliance on social information
    model_social = smf.logit(
        formula="social_info ~ age + C(site) + gender", data=df_social
    ).fit(disp=False)

    # Model 2: majority preference among social learners
    model_majority = smf.logit(
        formula="majority_choice ~ age + C(site) + gender", data=df_majority
    ).fit(disp=False)

    # Extract p-values for age and site indicators
    pvals_social = model_social.pvalues
    p_age_social = float(pvals_social.get("age", np.nan))
    p_site_social = float(
        pvals_social[[ix for ix in pvals_social.index if ix.startswith("C(site)")]]
        .min()
    )

    pvals_majority = model_majority.pvalues
    p_age_majority = float(pvals_majority.get("age", np.nan))
    site_terms_majority = [
        ix for ix in pvals_majority.index if ix.startswith("C(site)")
    ]
    p_site_majority = float(pvals_majority[site_terms_majority].min())

    # Simple age-binned descriptives for interpretability
    bins = [4, 6, 8, 10, 12, 14]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-13"]
    df_social["age_band"] = pd.cut(df_social["age"], bins=bins, labels=labels, right=False)
    df_majority["age_band"] = pd.cut(
        df_majority["age"], bins=bins, labels=labels, right=False
    )
    social_by_age = df_social.groupby("age_band")["social_info"].mean()
    majority_by_age = df_majority.groupby("age_band")["majority_choice"].mean()

    results = {
        "overall_social_rate": overall_social_rate,
        "overall_majority_rate": overall_majority_rate,
        "social_by_site": social_by_site.to_dict(),
        "majority_by_site": majority_by_site.to_dict(),
        "social_by_age": social_by_age.to_dict(),
        "majority_by_age": majority_by_age.to_dict(),
        "p_age_social": p_age_social,
        "p_site_social_min": p_site_social,
        "p_age_majority": p_age_majority,
        "p_site_majority_min": p_site_majority,
    }

    # Persist numeric results for inspection
    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

