import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
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

    # Derived variables
    df["site"] = df["site"].astype("category")

    # Reliance on social information: choosing either majority or minority option
    df["social_reliance"] = df["outcome"].isin([2, 3]).astype(int)

    # Among those using social info, preference for majority over minority
    df_social = df[df["social_reliance"] == 1].copy()
    df_social["majority_choice"] = (df_social["outcome"] == 2).astype(int)

    # Descriptive statistics
    outcome_counts = df["outcome"].value_counts(normalize=True).sort_index()
    social_rate = df["social_reliance"].mean()
    majority_rate = df_social["majority_choice"].mean()

    print("Overall outcome proportions (1=undemonstrated, 2=majority, 3=minority):")
    for k, v in outcome_counts.items():
        print(f"  {k}: {v:.3f}")
    print(f"\nOverall reliance on social information (2 or 3): {social_rate:.3f}")
    print(
        f"Among social choices, proportion following majority (2 vs 3): "
        f"{majority_rate:.3f}"
    )

    # Logistic regression: social_reliance ~ age + site
    model_social = smf.logit("social_reliance ~ age + C(site)", data=df).fit(disp=False)
    params_social = model_social.params
    pvalues_social = model_social.pvalues

    age_coef_social = float(params_social["age"])
    age_p_social = float(pvalues_social["age"])
    site_pvalues_social = {
        name: float(p)
        for name, p in pvalues_social.items()
        if name.startswith("C(site)")
    }
    any_site_sig_social = any(p < 0.05 for p in site_pvalues_social.values())

    # Logistic regression: majority_choice ~ age + site (among social learners)
    model_major = smf.logit("majority_choice ~ age + C(site)", data=df_social).fit(
        disp=False
    )
    params_major = model_major.params
    pvalues_major = model_major.pvalues

    age_coef_major = float(params_major["age"])
    age_p_major = float(pvalues_major["age"])
    site_pvalues_major = {
        name: float(p)
        for name, p in pvalues_major.items()
        if name.startswith("C(site)")
    }
    any_site_sig_major = any(p < 0.05 for p in site_pvalues_major.values())

    # Age effect as change in predicted probability from younger to older ages
    def avg_pred(model, data, age_value: float) -> float:
        tmp = data.copy()
        tmp["age"] = age_value
        return float(model.predict(tmp).mean())

    young_age = 6
    old_age = 12

    social_young = avg_pred(model_social, df, young_age)
    social_old = avg_pred(model_social, df, old_age)
    majority_young = avg_pred(model_major, df_social, young_age)
    majority_old = avg_pred(model_major, df_social, old_age)

    # Site-level variation (descriptive)
    site_social = df.groupby("site")["social_reliance"].mean()
    site_majority = df_social.groupby("site")["majority_choice"].mean()

    social_site_range = float(site_social.max() - site_social.min())
    majority_site_range = float(site_majority.max() - site_majority.min())

    print("\nLogistic regression: social_reliance ~ age + site")
    print(f"  age coef: {age_coef_social:.3f}, p={age_p_social:.4g}")
    print(
        f"  any site term p<0.05: {any_site_sig_social} "
        f"(min site p={min(site_pvalues_social.values()):.4g})"
    )
    print(
        f"  predicted social reliance at age {young_age}: {social_young:.3f}, "
        f"at age {old_age}: {social_old:.3f}"
    )
    print(f"  site-wise social reliance range: {social_site_range:.3f}")

    print("\nLogistic regression: majority_choice ~ age + site (social learners only)")
    print(f"  age coef: {age_coef_major:.3f}, p={age_p_major:.4g}")
    print(
        f"  any site term p<0.05: {any_site_sig_major} "
        f"(min site p={min(site_pvalues_major.values()):.4g})"
    )
    print(
        f"  predicted majority preference at age {young_age}: {majority_young:.3f}, "
        f"at age {old_age}: {majority_old:.3f}"
    )
    print(f"  site-wise majority preference range: {majority_site_range:.3f}")

    # Also dump a compact JSON summary in case it is helpful later.
    summary = {
        "overall": {
            "outcome_props": outcome_counts.to_dict(),
            "social_rate": social_rate,
            "majority_rate_given_social": majority_rate,
        },
        "models": {
            "social_reliance": {
                "age_coef": age_coef_social,
                "age_p": age_p_social,
                "any_site_p_lt_0_05": any_site_sig_social,
                "min_site_p": min(site_pvalues_social.values()),
                "pred_prob_young": social_young,
                "pred_prob_old": social_old,
                "site_range": social_site_range,
            },
            "majority_preference": {
                "age_coef": age_coef_major,
                "age_p": age_p_major,
                "any_site_p_lt_0_05": any_site_sig_major,
                "min_site_p": min(site_pvalues_major.values()),
                "pred_prob_young": majority_young,
                "pred_prob_old": majority_old,
                "site_range": majority_site_range,
            },
        },
    }

    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote analysis_summary.json with key statistics.")


if __name__ == "__main__":
    main()

