import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables:
    # - Student–teacher ratio = total enrollment / number of teachers
    # - Academic performance = mean of reading and math scores
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["performance"] = df[["feature14", "feature15"]].mean(axis=1)

    # Basic sanity checks
    print("Descriptive statistics for key variables:")
    print(df[["student_teacher_ratio", "performance"]].describe(), end="\n\n")

    # Pairwise Pearson correlation
    r, p = stats.pearsonr(df["student_teacher_ratio"], df["performance"])
    print(f"Pearson correlation (STR vs performance): r = {r:.3f}, p = {p:.3g}")

    # Simple OLS: performance ~ STR
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["performance"], X_simple).fit()
    print("\nSimple OLS regression: performance ~ STR")
    print(model_simple.summary())

    # Multiple OLS controlling for key demographics and resources
    controls = [
        "feature12",  # district average income
        "feature13",  # percent English learners
        "feature9",   # percent qualifying for reduced-price lunch
        "feature11",  # expenditure per student
    ]
    X_controls = sm.add_constant(df[["student_teacher_ratio"] + controls])
    model_controls = sm.OLS(df["performance"], X_controls).fit()
    print("\nMultiple OLS regression: performance ~ STR + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

