import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meaning using info.json descriptions.
    enrollment = df["english"]  # Total enrollment (students)
    teachers = df["students"]  # Number of teachers
    read_score = df["district"]  # Average reading score
    math_score = df["expenditure"]  # Average math score

    # Construct key variables
    df["stratio"] = enrollment / teachers
    df["avg_score"] = (read_score + math_score) / 2

    # Basic sanity checks
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe(), "\n")

    # Simple correlations with average score
    r_read, p_read = stats.pearsonr(df["stratio"], read_score)
    r_math, p_math = stats.pearsonr(df["stratio"], math_score)
    r_avg, p_avg = stats.pearsonr(df["stratio"], df["avg_score"])

    print("Pearson correlations (ratio vs scores):")
    print(f"  Read: r={r_read:.3f}, p={p_read:.4g}")
    print(f"  Math: r={r_math:.3f}, p={p_math:.4g}")
    print(f"  Avg : r={r_avg:.3f}, p={p_avg:.4g}\n")

    # Bivariate OLS: avg_score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    print("Bivariate OLS: avg_score ~ stratio")
    print(model_simple.summary(), "\n")

    # Multivariate OLS controlling for income and demographics
    # income: district average income
    # school: percent qualifying for CalWorks (income assistance)
    # computer: percent qualifying for reduced-price lunch
    # rownames: percent of English learners
    controls = df[["income", "school", "computer", "rownames"]]
    X_multi = sm.add_constant(pd.concat([df["stratio"], controls], axis=1))
    model_multi = sm.OLS(df["avg_score"], X_multi).fit()
    print("Multivariate OLS: avg_score ~ stratio + controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

