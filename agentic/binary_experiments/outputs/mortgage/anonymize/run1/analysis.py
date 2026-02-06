import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("mortgage.csv")

    # Outcome: accepted (1) vs denied (0)
    y = df["feature14"]

    # Unadjusted approval rates by gender
    approval_rates = df.groupby("feature2")["feature14"].mean()
    # 0 = male, 1 = female
    unadjusted_diff = approval_rates.get(1.0, float('nan')) - approval_rates.get(0.0, float('nan'))

    # Adjusted logistic regression with controls
    # Exclude feature11 (denied) to avoid perfect collinearity with accepted
    control_cols = [
        "feature1",
        "feature3",
        "feature4",
        "feature5",
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
        "feature12",
        "feature13",
    ]
    X = df[["feature2"] + control_cols]
    X = sm.add_constant(X, has_constant="add")

    # Drop rows with missing or non-finite values
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    coef = result.params["feature2"]
    pval = result.pvalues["feature2"]

    # Save key outputs for conclusion
    print("Unadjusted approval rates (male=0, female=1):")
    print(approval_rates)
    print(f"Unadjusted approval rate difference (female - male): {unadjusted_diff:.4f}")
    print("\nAdjusted logistic regression (accepted ~ female + controls):")
    print(f"Female coef (log-odds): {coef:.4f}")
    print(f"Female p-value: {pval:.6f}")


if __name__ == "__main__":
    main()
