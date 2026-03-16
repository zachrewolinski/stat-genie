import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    if not data_path.exists():
        raise FileNotFoundError("boxes.csv not found in current directory.")

    df = pd.read_csv(data_path)
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
    df["social_reliance"] = (df["outcome"] != 1).astype(int)
    df["majority_choice"] = np.where(
        df["outcome"].isin([2, 3]), (df["outcome"] == 2).astype(int), np.nan
    )

    # Basic descriptives
    print("N observations:", len(df))
    print("Sites:", sorted(df["site"].unique()))
    print("Age range:", df["age"].min(), "to", df["age"].max())
    print()

    # Logistic regression: reliance on any social information
    model_social = smf.logit("social_reliance ~ age + C(site)", data=df).fit(disp=False)
    print("=== Social reliance model (any demonstrated option vs undemonstrated) ===")
    print(model_social.summary())
    p_age_social = float(model_social.pvalues["age"])

    # Average predicted probabilities at younger vs older ages
    for age in [4, 8, 12]:
        df_age = df.copy()
        df_age["age"] = age
        prob = float(model_social.predict(df_age).mean())
        print(f"Mean predicted social reliance at age {age}: {prob:.3f}")

    site_terms_social = model_social.pvalues.filter(like="C(site)")
    any_site_social = bool((site_terms_social < 0.05).any())

    print()

    # Logistic regression: preference for majority vs minority, among social learners
    df_majority = df.dropna(subset=["majority_choice"]).copy()
    model_majority = smf.logit(
        "majority_choice ~ age + C(site)", data=df_majority
    ).fit(disp=False)
    print("=== Majority preference model (majority vs minority) ===")
    print(model_majority.summary())
    p_age_majority = float(model_majority.pvalues["age"])

    for age in [4, 8, 12]:
        dfm_age = df_majority.copy()
        dfm_age["age"] = age
        prob = float(model_majority.predict(dfm_age).mean())
        print(f"Mean predicted majority choice at age {age}: {prob:.3f}")

    site_terms_majority = model_majority.pvalues.filter(like="C(site)")
    any_site_majority = bool((site_terms_majority < 0.05).any())

    # Simple JSON summary of key statistics (not the final conclusion file)
    summary = {
        "p_age_social": p_age_social,
        "any_site_social_p_lt_0_05": any_site_social,
        "p_age_majority": p_age_majority,
        "any_site_majority_p_lt_0_05": any_site_majority,
    }
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("Wrote key statistics to analysis_summary.json")


if __name__ == "__main__":
    main()

