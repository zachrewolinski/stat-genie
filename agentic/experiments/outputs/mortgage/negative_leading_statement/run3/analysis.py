import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = "mortgage.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic approval rates by gender
    approval_by_gender = df.groupby("female")["accept"].mean()
    counts_by_gender = df.groupby("female")["accept"].count()
    successes_by_gender = df.groupby("female")["accept"].sum()

    # Two-proportion z-test for difference in approval rates
    count = np.array([successes_by_gender.get(1.0, 0), successes_by_gender.get(0.0, 0)])
    nobs = np.array([counts_by_gender.get(1.0, 0), counts_by_gender.get(0.0, 0)])
    zstat, pval = proportions_ztest(count, nobs, alternative="two-sided")

    # Logistic regression controlling for creditworthiness and other factors
    features = [
        "female",
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
        "denied_PMI",
    ]
    model_df = df[features + ["accept"]].replace([np.inf, -np.inf], np.nan).dropna()
    X = model_df[features]
    X = sm.add_constant(X)
    y = model_df["accept"]

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    female_coef = result.params["female"]
    female_pval = result.pvalues["female"]
    female_odds_ratio = float(np.exp(female_coef))

    print("Approval rate by gender (female=1, male=0):")
    print(approval_by_gender)
    print()
    print("Counts by gender:")
    print(counts_by_gender)
    print()
    print("Two-proportion z-test for approval rates (female vs male):")
    print(f"z-stat: {zstat:.4f}, p-value: {pval:.4g}")
    print()
    print("Logit model (accept ~ controls + female):")
    print(f"female coef: {female_coef:.4f}")
    print(f"female odds ratio: {female_odds_ratio:.4f}")
    print(f"female p-value: {female_pval:.4g}")


if __name__ == "__main__":
    main()
