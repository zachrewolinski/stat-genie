import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to info.json metadata (not the original CASchools names):
    # - "english" currently holds total enrollment ("Total enrollment.")
    # - "students" currently holds number of teachers ("Number of teachers.")
    # - "district" is the average reading score.
    # - "expenditure" is the average math score.
    #
    # We construct:
    #   - student–teacher ratio = enrollment / teachers
    #   - academic performance = mean of reading and math scores

    # Guard against any zero or missing teacher counts
    df = df.copy()
    df = df[(df["students"] > 0) & df["english"].notna() & df["students"].notna()]

    df["stratio"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop any remaining missing values in the variables of interest
    df = df[["stratio", "testscr"]].dropna()

    # Simple linear regression: testscr ~ stratio
    X = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model = sm.OLS(y, X).fit()

    coef = model.params["stratio"]
    p_value = model.pvalues["stratio"]
    r_squared = model.rsquared

    # We expect that *lower* stratio (fewer students per teacher)
    # is associated with *higher* test scores.
    #
    # A negative coefficient on stratio supports the research hypothesis.

    # Map evidence strength to a Likert-style score in [-100, 100].
    # This mapping is heuristic but monotone in effect size and significance.
    if p_value < 0.001 and coef < 0:
        scalar = 90
    elif p_value < 0.01 and coef < 0:
        scalar = 70
    elif p_value < 0.05 and coef < 0:
        scalar = 50
    elif p_value < 0.1 and coef < 0:
        scalar = 30
    elif coef < 0:
        # Negative but not statistically strong
        scalar = 10
    elif p_value < 0.05 and coef > 0:
        # Statistically significant in the opposite direction
        scalar = -50
    else:
        # Weak or no clear pattern
        scalar = 0

    # As an additional consistency check, down-weight if the model fit is very poor.
    if r_squared < 0.02:
        scalar = int(round(scalar * 0.5))

    # Ensure scalar stays within [-100, 100]
    scalar = max(-100, min(100, int(round(scalar))))

    # Write scalar to the required output file with no extra text.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

