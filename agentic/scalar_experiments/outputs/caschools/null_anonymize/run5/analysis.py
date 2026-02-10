import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Rename only the columns we need based on info.json metadata
    df = df.rename(
        columns={
            "feature6": "enroll",   # Total enrollment
            "feature7": "teachers", # Number of teachers
            "feature8": "calwpct",  # % on CalWorks
            "feature9": "mealpct",  # % on reduced-price lunch
            "feature11": "expnstu", # Expenditure per student
            "feature12": "avginc",  # District average income (in $1,000)
            "feature13": "elpct",   # % of English learners
            "feature14": "readscr", # Average reading score
            "feature15": "mathscr", # Average math score
        }
    )

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["enroll"] / df["teachers"]
    df["testscr"] = df[["readscr", "mathscr"]].mean(axis=1)

    # Basic association: Pearson correlation between STR and test scores
    corr = df["stratio"].corr(df["testscr"])

    # Regression controlling for key demographics and resources
    covariates = ["stratio", "avginc", "elpct", "mealpct", "expnstu"]
    reg_df = df[covariates + ["testscr"]].dropna()
    X = sm.add_constant(reg_df[covariates])
    y = reg_df["testscr"]

    model = sm.OLS(y, X).fit()
    coef_str = model.params["stratio"]
    p_str = model.pvalues["stratio"]
    r2 = model.rsquared

    # Interpretation for logs (not used in final scalar file)
    print("Correlation between STR and testscr:", corr)
    print("OLS coef on STR:", coef_str)
    print("p-value for STR:", p_str)
    print("R-squared:", r2)

    # Map evidence to Likert scale (-100 to 100)
    # Lower STR (fewer students per teacher) associated with higher performance
    # corresponds to a NEGATIVE relationship between STR and testscr.
    # Use both correlation magnitude and regression significance.

    # Directional support: negative correlation and negative coefficient
    direction_agrees = (corr < 0) and (coef_str < 0)

    # Effect size component from correlation (cap at |corr| = 0.4)
    corr_strength = min(abs(corr) / 0.4, 1.0)

    # Significance bonus from regression p-value
    if p_str < 0.001:
        sig_bonus = 0.3
    elif p_str < 0.01:
        sig_bonus = 0.2
    elif p_str < 0.05:
        sig_bonus = 0.1
    else:
        sig_bonus = 0.0

    base_support = corr_strength + sig_bonus
    base_support = max(0.0, min(base_support, 1.0))

    # Convert to Likert score; positive means "Yes, there is an association"
    score = int(round(base_support * 100))

    if not direction_agrees:
        # Evidence contradicts the hypothesized direction
        score = -score

    # Ensure the score lies in [-100, 100]
    score = max(-100, min(100, score))

    # Log final interpretation
    print("Final Likert-style scalar conclusion:", score)

    # Write scalar to conclusion.txt with no extra text
    with open("conclusion.txt", "w") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()

