import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student–teacher ratio.
    df["str"] = df["students"] / df["teachers"]

    # Outcome: average of reading and math scores as overall performance.
    df["score"] = df[["read", "math"]].mean(axis=1)

    # Basic sanity checks: drop any implausible or missing values.
    df = df.replace([float("inf"), -float("inf")], pd.NA).dropna(
        subset=["str", "score"]
    )

    # Add key demographic and resource controls using available columns.
    controls = ["calworks", "lunch", "income", "english", "computer", "expenditure"]
    available_controls = [c for c in controls if c in df.columns]

    X = df[["str"] + available_controls]
    X = sm.add_constant(X)
    y = df["score"]

    model = sm.OLS(y, X).fit()

    # Extract the coefficient and p-value for student–teacher ratio.
    coef = model.params["str"]
    pval = model.pvalues["str"]

    # Map evidence strength to a Likert-style scalar in [-100, 100].
    # Lower ratio (more teachers per student) is expected to increase scores,
    # so a negative coefficient supports the researcher's "Yes" belief.
    if pval > 0.10:
        scalar = 0
    else:
        # Start from direction: negative is good (supports the hypothesis).
        direction = -1 if coef < 0 else 1

        # Normalize magnitude using a soft saturation on effect size per 10 students.
        effect_per_10 = coef * 10.0
        # Typical test-score SD is ~18–20; use 20 for scaling.
        standardized = abs(effect_per_10) / 20.0

        # Convert to [0, 1) using a bounded transformation, then to [0, 100].
        strength = 100.0 * (standardized / (1.0 + standardized))

        # Tighten scale to reflect statistical significance level.
        if pval < 0.001:
            sig_multiplier = 1.0
        elif pval < 0.01:
            sig_multiplier = 0.85
        elif pval < 0.05:
            sig_multiplier = 0.7
        else:  # p <= 0.10
            sig_multiplier = 0.4

        scalar = direction * strength * sig_multiplier

    # Round to nearest integer and clip into [-100, 100].
    scalar_int = int(round(max(-100, min(100, scalar))))

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(scalar_int), encoding="utf-8")


if __name__ == "__main__":
    main()

