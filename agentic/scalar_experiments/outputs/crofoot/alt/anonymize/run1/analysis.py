import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def fit_logit(y, X, label: str) -> None:
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print(f"\n=== Logistic regression: {label} ===")
    print("Coefficients:")
    print(result.params)
    print("\nStandard errors:")
    print(result.bse)
    print("\nP-values:")
    print(result.pvalues)
    print("Pseudo R-squared:", result.prsquared)


def chi_square_test(table: pd.DataFrame, label: str) -> None:
    chi2, p, dof, exp = stats.chi2_contingency(table)
    print(f"\n=== Chi-square test: {label} ===")
    print("Contingency table:")
    print(table)
    print(f"Chi2 = {chi2:.3f}, df = {dof}, p-value = {p:.4f}")


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"]

    # Relative group size: focal group size minus other group size
    df["rel_group_size"] = df["feature7"] - df["feature8"]
    df["focal_larger"] = (df["feature7"] > df["feature8"]).astype(int)

    # Contest location advantage for focal group:
    # distance_other_from_core - distance_focal_from_core
    # Positive values mean the focal group is closer to its home-range center.
    df["location_advantage"] = df["feature6"] - df["feature5"]
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    # Logistic regression with continuous predictors
    fit_logit(
        y,
        df[["rel_group_size", "location_advantage"]],
        label="continuous rel_group_size and location_advantage",
    )

    # Logistic regression with binary indicators
    fit_logit(
        y,
        df[["focal_larger", "focal_closer"]],
        label="binary focal_larger and focal_closer",
    )

    # Simple chi-square tests for association
    chi_square_test(
        pd.crosstab(df["focal_larger"], y),
        label="focal_larger vs focal_win",
    )
    chi_square_test(
        pd.crosstab(df["focal_closer"], y),
        label="focal_closer (home-field) vs focal_win",
    )


if __name__ == "__main__":
    main()

