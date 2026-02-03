import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def main():
    df = pd.read_csv("mortgage.csv")

    # Basic acceptance rates by gender
    rates = df.groupby("female")["accept"].mean().rename({0.0: "male", 1.0: "female"})
    counts = df.groupby("female")["accept"].agg(["sum", "count"]).rename(index={0.0: "male", 1.0: "female"})

    # Two-proportion z-test for acceptance rate difference
    successes = counts["sum"].values
    nobs = counts["count"].values
    zstat, pval = proportions_ztest(successes, nobs, alternative="two-sided")

    # Logistic regression controlling for other covariates
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
    X = df[features]
    X = sm.add_constant(X)
    y = df["accept"]

    # Remove rows with missing or infinite values for modeling
    model_data = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y_model = model_data["accept"]
    X_model = model_data.drop(columns=["accept"])

    model = sm.Logit(y_model, X_model)
    result = model.fit(disp=False)

    coef_female = result.params["female"]
    p_female = result.pvalues["female"]
    odds_ratio = float(np.exp(coef_female))

    print("Acceptance rates by gender (mean accept):")
    print(rates)
    print("\nCounts (accepts / total):")
    print(counts)
    print("\nTwo-proportion z-test (acceptance rate difference):")
    print(f"z = {zstat:.3f}, p = {pval:.4g}")
    print("\nLogistic regression (accept ~ gender + controls):")
    print(f"female coef = {coef_female:.4f}, odds ratio = {odds_ratio:.4f}, p = {p_female:.4g}")


if __name__ == "__main__":
    main()
