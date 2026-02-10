import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: higher values mean larger classes.
    df["stratio"] = df["students"] / df["teachers"]

    # Basic correlations with achievement.
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])

    # Control for key covariates that clearly relate to achievement.
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    X = df[["stratio"] + controls].copy()
    X = sm.add_constant(X)

    y_read = df["read"]
    y_math = df["math"]

    model_read = sm.OLS(y_read, X).fit()
    model_math = sm.OLS(y_math, X).fit()

    coef_read = model_read.params["stratio"]
    p_read = model_read.pvalues["stratio"]

    coef_math = model_math.params["stratio"]
    p_math = model_math.pvalues["stratio"]

    # Interpret results: negative coefficient means lower ratio -> higher scores.
    # We'll create a qualitative score in [-100, 100], where positive numbers
    # indicate evidence that lower student–teacher ratios are associated with
    # higher achievement.
    evidence_score = 0

    for coef, p, corr in [
        (coef_read, p_read, corr_read),
        (coef_math, p_math, corr_math),
    ]:
        if pd.isna(coef) or pd.isna(p):
            continue

        direction = -1 if coef > 0 else 1  # negative coef supports "Yes"

        # Strength from significance and correlation magnitude.
        if p < 0.001:
            base = 40
        elif p < 0.01:
            base = 30
        elif p < 0.05:
            base = 20
        elif p < 0.1:
            base = 10
        else:
            base = 5

        strength = base * min(1.0, abs(corr) / 0.3)
        evidence_score += direction * strength

    # Average across reading and math.
    if evidence_score != 0:
        evidence_score /= 2.0

    # Clip to [-100, 100] and round to nearest integer.
    final_scalar = int(max(-100, min(100, round(evidence_score))))

    # Write only the scalar to conclusion.txt as required.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(final_scalar))


if __name__ == "__main__":
    main()

