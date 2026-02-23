import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site_id",
        }
    )

    # Derived variables
    df["social_choice"] = df["choice"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["choice"] == 2).astype(int)
    df["site"] = df["site_id"].astype("category")

    # Age groups (rough developmental stages)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[4, 6, 8, 10, 12, 14],
        include_lowest=True,
        right=True,
    )

    # Overall reliance on social information and majority bias
    social_rate = df["social_choice"].mean()
    majority_among_social = df.loc[df["social_choice"] == 1, "majority_choice"].mean()

    # Variation across cultures (sites)
    social_by_site = df.groupby("site")["social_choice"].mean()
    majority_by_site = (
        df[df["social_choice"] == 1].groupby("site")["majority_choice"].mean()
    )

    # Variation across developmental stages (age groups)
    social_by_age_group = df.groupby("age_group")["social_choice"].mean()
    majority_by_age_group = df[df["social_choice"] == 1].groupby("age_group")[
        "majority_choice"
    ].mean()

    # Logistic regressions to test for age and cultural effects
    model_social = smf.logit("social_choice ~ age + C(site)", data=df).fit(disp=False)
    model_majority = smf.logit(
        "majority_choice ~ age + C(site)", data=df[df["social_choice"] == 1]
    ).fit(disp=False)

    # Collect key statistics for manual interpretation
    results = {
        "n": int(len(df)),
        "social_rate_overall": float(social_rate),
        "majority_among_social_overall": float(majority_among_social),
        "social_by_site": {str(k): float(v) for k, v in social_by_site.items()},
        "majority_by_site": {
            str(k): float(v) for k, v in majority_by_site.items()
        },
        "social_by_age_group": {
            str(k): float(v) for k, v in social_by_age_group.items()
        },
        "majority_by_age_group": {
            str(k): float(v) for k, v in majority_by_age_group.items()
        },
        "logit_social_params": model_social.params.to_dict(),
        "logit_social_pvalues": model_social.pvalues.to_dict(),
        "logit_majority_params": model_majority.params.to_dict(),
        "logit_majority_pvalues": model_majority.pvalues.to_dict(),
    }

    # Print JSON to stdout so it can be inspected from the CLI
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
