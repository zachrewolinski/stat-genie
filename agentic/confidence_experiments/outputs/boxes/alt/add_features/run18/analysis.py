import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Encode outcomes
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    df["culture"] = df["culture"].astype("category")

    print("=== Basic counts ===")
    print("N:", len(df))
    print("y distribution:")
    print(df["y"].value_counts().sort_index())
    print("\nSocial (chose demonstrated option) rate:", df["social"].mean())
    print("Majority-choice rate among social choosers:", df.loc[df["social"] == 1, "majority_choice"].mean())

    print("\n=== Social learning (any demonstrated option vs undemonstrated) ===")
    social_model = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    print(social_model.summary())

    # Age effect: difference in predicted probabilities from 25th to 75th percentile
    age_q25, age_q75 = df["age"].quantile([0.25, 0.75])
    base_culture = df["culture"].mode()[0]
    grid_social = pd.DataFrame(
        {
            "age": [age_q25, age_q75],
            "culture": [base_culture, base_culture],
        }
    )
    pred_social = social_model.predict(grid_social)
    print("\nPredicted social probability at age 25th pct:", float(pred_social.iloc[0]))
    print("Predicted social probability at age 75th pct:", float(pred_social.iloc[1]))

    social_by_age = df.groupby("age")["social"].mean()
    social_by_culture = df.groupby("culture")["social"].mean()
    print("\nSocial rate by age:")
    print(social_by_age)
    print("\nSocial rate by culture:")
    print(social_by_culture)

    print("\nCulture coefficients (social model):")
    for name, coef, p in zip(
        social_model.params.index, social_model.params.values, social_model.pvalues.values
    ):
        if name.startswith("C(culture)"):
            print(f"{name}: coef={coef:.3f}, p={p:.4g}")

    print("\nAge effect (social model):")
    print(f"coef={social_model.params['age']:.3f}, p={social_model.pvalues['age']:.4g}")

    print("\n=== Majority preference (majority vs minority among social choosers) ===")
    df_pref = df[df["social"] == 1].copy()
    pref_model = smf.logit("majority_choice ~ age + C(culture)", data=df_pref).fit(disp=False)
    print(pref_model.summary())

    grid_pref = pd.DataFrame(
        {
            "age": [age_q25, age_q75],
            "culture": [base_culture, base_culture],
        }
    )
    pred_pref = pref_model.predict(grid_pref)
    print("\nPredicted majority-choice probability at age 25th pct:", float(pred_pref.iloc[0]))
    print("Predicted majority-choice probability at age 75th pct:", float(pred_pref.iloc[1]))

    pref_by_age = df_pref.groupby("age")["majority_choice"].mean()
    pref_by_culture = df_pref.groupby("culture")["majority_choice"].mean()
    print("\nMajority-choice rate by age (social choosers):")
    print(pref_by_age)
    print("\nMajority-choice rate by culture (social choosers):")
    print(pref_by_culture)

    print("\nCulture coefficients (majority-preference model):")
    for name, coef, p in zip(pref_model.params.index, pref_model.params.values, pref_model.pvalues.values):
        if name.startswith("C(culture)"):
            print(f"{name}: coef={coef:.3f}, p={p:.4g}")

    print("\nAge effect (majority-preference model):")
    print(f"coef={pref_model.params['age']:.3f}, p={pref_model.pvalues['age']:.4g}")

    # Additional checks treating age as categorical (developmental stages approximation)
    print("\n=== Social learning with age as categorical ===")
    social_age_cat = smf.logit("social ~ C(age)", data=df).fit(disp=False)
    print(social_age_cat.summary())

    print("\n=== Majority preference with age as categorical (social choosers) ===")
    pref_age_cat = smf.logit("majority_choice ~ C(age)", data=df_pref).fit(disp=False)
    print(pref_age_cat.summary())


if __name__ == "__main__":
    main()
