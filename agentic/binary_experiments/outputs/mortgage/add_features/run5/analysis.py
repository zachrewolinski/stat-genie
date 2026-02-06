import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Keep mortgage-relevant columns
    cols = [
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
        "accept",
    ]
    df = df[cols].copy()

    # Basic cleaning: drop rows with missing values in model variables
    df = df.dropna()

    # Descriptive: approval rates by gender
    approval_by_gender = df.groupby("female")["accept"].mean()

    # Logistic regression with controls
    y = df["accept"]
    X = df.drop(columns=["accept"])
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    female_coef = result.params.get("female", np.nan)
    female_p = result.pvalues.get("female", np.nan)
    female_or = np.exp(female_coef) if pd.notnull(female_coef) else np.nan

    # Save key outputs for the conclusion
    summary = {
        "n": int(len(df)),
        "approval_rate_male": float(approval_by_gender.get(0.0, np.nan)),
        "approval_rate_female": float(approval_by_gender.get(1.0, np.nan)),
        "female_coef": float(female_coef),
        "female_p": float(female_p),
        "female_odds_ratio": float(female_or),
    }

    # Write a small text output for transparency
    with open("analysis_results.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # Also write model summary for optional inspection
    with open("model_summary.txt", "w") as f:
        f.write(result.summary2().as_text())


if __name__ == "__main__":
    main()
