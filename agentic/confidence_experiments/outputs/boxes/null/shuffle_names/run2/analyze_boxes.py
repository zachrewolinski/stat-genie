import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Encode key outcomes
    # majority_first: 1 = undemonstrated, 2 = majority, 3 = minority
    df = df.copy()
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(df["majority_first"] == 2, 1, np.nan)
    df.loc[df["majority_first"] == 3, "majority_choice"] = 0

    # Center age for stability
    df["age_c"] = df["age"] - df["age"].mean()

    # Treat site id "y" as categorical culture proxy
    df["site"] = df["y"].astype("category")

    results = {}

    # Model 1: reliance on social information (social vs asocial)
    model_social = smf.glm(
        formula="social_choice ~ age_c + C(site) + age_c:C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    results["social_params"] = model_social.params.to_dict()
    results["social_pvalues"] = model_social.pvalues.to_dict()

    # Model 2: preference for majority vs minority, among social choices
    df_social = df[df["majority_first"].isin([2, 3])].copy()
    if df_social["majority_choice"].nunique() == 2:
        model_majority = smf.glm(
            formula="majority_choice ~ age_c + C(site) + age_c:C(site)",
            data=df_social,
            family=sm.families.Binomial(),
        ).fit()
        results["majority_params"] = model_majority.params.to_dict()
        results["majority_pvalues"] = model_majority.pvalues.to_dict()
    else:
        results["majority_params"] = {}
        results["majority_pvalues"] = {}

    # Basic descriptive summaries to aid interpretation
    social_rate_by_site = (
        df.groupby("site")["social_choice"].mean().to_dict()
    )
    majority_rate_by_site = (
        df_social.groupby("site")["majority_choice"].mean().to_dict()
    )

    results["social_rate_by_site"] = social_rate_by_site
    results["majority_rate_by_site"] = majority_rate_by_site

    # Summaries by age (quartiles) for additional descriptive insight
    df["age_group"] = pd.qcut(df["age"], q=4, duplicates="drop")
    social_by_age_group = (
        df.groupby("age_group")["social_choice"].mean().to_dict()
    )
    majority_by_age_group = (
        df_social.groupby(pd.qcut(df_social["age"], q=4, duplicates="drop"))[
            "majority_choice"
        ]
        .mean()
        .to_dict()
    )
    results["social_by_age_group"] = {
        str(k): float(v) for k, v in social_by_age_group.items()
    }
    results["majority_by_age_group"] = {
        str(k): float(v) for k, v in majority_by_age_group.items()
    }

    # Print JSON so the calling process can inspect statistical evidence
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
