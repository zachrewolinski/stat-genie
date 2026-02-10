import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio and academic performance
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["perf"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing or invalid values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["stratio", "perf"])

    # Basic correlation
    r = df["stratio"].corr(df["perf"])

    # Simple linear regression: perf ~ stratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["perf"], X).fit()
    slope = float(model.params["stratio"])
    p_value = float(model.pvalues["stratio"])

    # Map evidence to Likert scalar in [-100, 100]
    # Interpretation:
    # - Negative slope / correlation => lower STR associated with higher performance (evidence for "Yes").
    # - Positive slope / correlation => evidence against.
    # - Magnitude of effect and statistical significance control strength.

    # Start from correlation magnitude as primary effect size
    effect_strength = min(abs(r), 1.0)

    # Base score direction: negative r implies "Yes"
    base_score = effect_strength * 100.0
    if r < 0:
        score = base_score
    else:
        score = -base_score

    # Adjust for statistical significance of the slope
    # Strongly significant: multiply by 1.0
    # Moderately significant: down-weight
    # Weak / not significant: heavily shrink toward 0
    if p_value < 0.001:
        sig_weight = 1.0
    elif p_value < 0.01:
        sig_weight = 0.9
    elif p_value < 0.05:
        sig_weight = 0.75
    elif p_value < 0.1:
        sig_weight = 0.5
    else:
        sig_weight = 0.2

    score *= sig_weight

    # Clip to [-100, 100] and round to nearest integer
    scalar = int(np.clip(np.round(score), -100, 100))

    # Print a brief summary for debugging/inspection
    print("Correlation between student-teacher ratio and performance:", r)
    print("Regression slope (perf ~ stratio):", slope)
    print("p-value for slope:", p_value)
    print("Derived Likert scalar:", scalar)

    # Write the scalar conclusion to file as required
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

