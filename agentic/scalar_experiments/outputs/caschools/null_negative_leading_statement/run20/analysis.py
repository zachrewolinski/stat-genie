import math

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def compute_likert_scalar_from_correlation(r: float, p_value: float) -> int:
    """
    Map the evidence about the association between student–teacher ratio and performance
    to an integer in [-100, 100], where positive values indicate support for:
    "Lower student–teacher ratio is associated with higher academic performance."
    """
    r = float(r)
    p_value = float(p_value)

    # If correlation is effectively zero or not statistically meaningful, return neutral.
    if math.isnan(r) or math.isnan(p_value) or p_value >= 0.2 or abs(r) < 0.02:
        return 0

    # Evidence strength from p-value.
    if p_value < 1e-6:
        sig_strength = 1.0
    elif p_value < 1e-3:
        sig_strength = 0.8
    elif p_value < 1e-2:
        sig_strength = 0.6
    elif p_value < 5e-2:
        sig_strength = 0.4
    else:
        sig_strength = 0.2

    # Effect size component from |r| (cap at 0.5 for mapping).
    effect_strength = min(1.0, abs(r) / 0.5)

    combined_strength = sig_strength * effect_strength

    # Sign convention: negative correlation between STR and score => positive support for the hypothesis.
    sign = 1 if r < 0 else -1

    scalar = int(round(sign * combined_strength * 100))

    # Ensure scalar lies in [-100, 100].
    scalar = max(-100, min(100, scalar))
    return scalar


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Simple Pearson correlation between class size (student–teacher ratio) and performance.
    r, p_value = stats.pearsonr(df["stratio"], df["avgscore"])

    # Additional regression analysis controlling for observed covariates, for robustness.
    covariates = ["income", "english", "lunch"]
    available_covariates = [c for c in covariates if c in df.columns]
    if available_covariates:
        X = df[["stratio"] + available_covariates]
        X = sm.add_constant(X)
        y = df["avgscore"]
        ols_model = sm.OLS(y, X).fit()
        coef_stratio = float(ols_model.params["stratio"])
        p_stratio = float(ols_model.pvalues["stratio"])
        # Print a concise summary for human inspection if needed.
        print("Correlation r(stratio, avgscore):", r)
        print("Correlation p-value:", p_value)
        print("OLS coefficient on stratio:", coef_stratio)
        print("OLS p-value for stratio:", p_stratio)
    else:
        print("Correlation r(stratio, avgscore):", r)
        print("Correlation p-value:", p_value)

    scalar = compute_likert_scalar_from_correlation(r, p_value)

    # Write the scalar conclusion to the required file, as the only content.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

