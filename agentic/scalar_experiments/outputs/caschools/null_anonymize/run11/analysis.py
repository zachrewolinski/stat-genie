import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to the metadata:
    # feature6: total enrollment
    # feature7: number of teachers (FTE)
    # feature14: average reading score
    # feature15: average math score
    df["stud_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Drop any obviously problematic rows (e.g., zero or missing teachers)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["stud_teacher_ratio", "avg_score"]
    )

    # Simple Pearson correlation between ratio and performance
    corr = df["stud_teacher_ratio"].corr(df["avg_score"])

    # Simple bivariate OLS: avg_score ~ stud_teacher_ratio
    X_simple = sm.add_constant(df["stud_teacher_ratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()

    # Multivariate OLS controlling for key demographics & resources
    controls = [
        "feature8",   # % CalWorks (income assistance)
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district avg income (in $1,000s)
        "feature13",  # % English learners
    ]
    available_controls = [c for c in controls if c in df.columns]
    X_multi = sm.add_constant(df[["stud_teacher_ratio"] + available_controls])
    model_multi = sm.OLS(df["avg_score"], X_multi).fit()

    print("Number of districts used:", len(df))
    print("\n=== Correlation ===")
    print(f"Pearson corr(stud_teacher_ratio, avg_score) = {corr:.4f}")

    print("\n=== Simple OLS: avg_score ~ stud_teacher_ratio ===")
    print(model_simple.summary())

    print("\n=== Multivariate OLS with controls ===")
    print("Controls:", available_controls)
    print(model_multi.summary())


if __name__ == "__main__":
    main()

