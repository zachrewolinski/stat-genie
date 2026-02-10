import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social reliance: chose either demonstrated option (majority or minority)
    df["social_reliance"] = df["majority_first"].isin([2, 3]).astype(int)

    # Majority preference: among socially reliant children, chose the majority option
    reliant = df[df["social_reliance"] == 1].copy()
    reliant["majority_preference"] = (reliant["majority_first"] == 2).astype(int)

    # Logistic models via GLM with binomial family for stability
    sr_model = smf.glm(
        "social_reliance ~ age + C(y)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    mp_model = smf.glm(
        "majority_preference ~ age + C(y)",
        data=reliant,
        family=sm.families.Binomial(),
    ).fit()

    # Simple significance flags
    sr_pvalues = sr_model.pvalues
    mp_pvalues = mp_model.pvalues

    sr_age_sig = float(sr_pvalues.get("age", 1.0)) < 0.05
    mp_age_sig = float(mp_pvalues.get("age", 1.0)) < 0.05

    sr_culture_sig = any(
        float(p) < 0.05 for name, p in sr_pvalues.items() if name.startswith("C(y)[T.")
    )
    mp_culture_sig = any(
        float(p) < 0.05 for name, p in mp_pvalues.items() if name.startswith("C(y)[T.")
    )

    print("Social reliance model summary:")
    print(sr_model.summary())
    print("\nMajority preference model summary:")
    print(mp_model.summary())

    print("\nSignificance flags (p < 0.05):")
    print(f"Social reliance ~ age: {sr_age_sig}")
    print(f"Social reliance ~ culture (sites): {sr_culture_sig}")
    print(f"Majority preference ~ age: {mp_age_sig}")
    print(f"Majority preference ~ culture (sites): {mp_culture_sig}")


if __name__ == "__main__":
    main()

