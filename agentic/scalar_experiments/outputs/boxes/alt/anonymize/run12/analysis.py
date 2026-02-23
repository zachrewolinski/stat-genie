import json

import pandas as pd
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Rename columns with meaningful labels
    df = df.rename(
        columns={
            "feature1": "outcome",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Derived variables
    df["social"] = df["outcome"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["outcome"] == 2).astype(int)
    df["minority_choice"] = (df["outcome"] == 3).astype(int)

    # Age bands for clearer developmental stages
    bins = [3, 6, 8, 10, 12, 15]
    labels = ["4-6", "7-8", "9-10", "11-12", "13-14"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

    # Overall descriptives
    overall_social_rate = df["social"].mean()
    overall_majority_rate = df.loc[df["social"] == 1, "majority_choice"].mean()

    # Social reliance by site and age band
    social_by_site = pd.crosstab(df["site"], df["social"])
    social_by_age = pd.crosstab(df["age_band"], df["social"])

    # Majority preference among social learners
    social_learners = df[df["social"] == 1].copy()
    majority_by_site = pd.crosstab(social_learners["site"], social_learners["majority_choice"])
    majority_by_age = pd.crosstab(
        social_learners["age_band"],
        social_learners["majority_choice"],
    )

    # Chi-square tests
    chi2_site_social, p_site_social, _, _ = stats.chi2_contingency(social_by_site)
    chi2_age_social, p_age_social, _, _ = stats.chi2_contingency(social_by_age)

    chi2_site_majority, p_site_majority, _, _ = stats.chi2_contingency(majority_by_site)
    chi2_age_majority, p_age_majority, _, _ = stats.chi2_contingency(majority_by_age)

    # Proportions and ranges
    social_site_props = social_by_site.div(social_by_site.sum(axis=1), axis=0)[1]
    social_age_props = social_by_age.div(social_by_age.sum(axis=1), axis=0)[1]

    majority_site_props = majority_by_site.div(majority_by_site.sum(axis=1), axis=0)[1]
    majority_age_props = majority_by_age.div(majority_by_age.sum(axis=1), axis=0)[1]

    results = {
        "n": int(len(df)),
        "overall_social_rate": float(overall_social_rate),
        "overall_majority_rate_among_social": float(overall_majority_rate),
        "social_by_site": social_site_props.to_dict(),
        "social_by_age_band": social_age_props.to_dict(),
        "majority_by_site": majority_site_props.to_dict(),
        "majority_by_age_band": majority_age_props.to_dict(),
        "chi2": {
            "site_social": {"chi2": float(chi2_site_social), "p": float(p_site_social)},
            "age_social": {"chi2": float(chi2_age_social), "p": float(p_age_social)},
            "site_majority": {
                "chi2": float(chi2_site_majority),
                "p": float(p_site_majority),
            },
            "age_majority": {
                "chi2": float(chi2_age_majority),
                "p": float(p_age_majority),
            },
        },
        "ranges": {
            "social_site_range": float(social_site_props.max() - social_site_props.min()),
            "social_age_range": float(social_age_props.max() - social_age_props.min()),
            "majority_site_range": float(
                majority_site_props.max() - majority_site_props.min()
            ),
            "majority_age_range": float(
                majority_age_props.max() - majority_age_props.min()
            ),
        },
    }

    # Print as JSON so the agent can inspect results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

