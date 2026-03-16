import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["students_per_teacher"] = df["feature6"] / df["feature7"]
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    print("Descriptive statistics:")
    print("\nStudents per teacher:")
    print(df["students_per_teacher"].describe())

    print("\nAverage test score (reading + math) / 2:")
    print(df["avg_score"].describe())

    # Bivariate correlation
    corr, p_corr = stats.pearsonr(df["students_per_teacher"], df["avg_score"])
    print(
        f"\nPearson correlation (students_per_teacher vs avg_score): "
        f"{corr:.3f} (p-value = {p_corr:.3g})"
    )

    for col in ["feature14", "feature15"]:
        corr_subj, p_subj = stats.pearsonr(df["students_per_teacher"], df[col])
        print(
            f"Pearson correlation (students_per_teacher vs {col}): "
            f"{corr_subj:.3f} (p-value = {p_subj:.3g})"
        )

    # Simple OLS regression: avg_score ~ students_per_teacher
    X_simple = sm.add_constant(df["students_per_teacher"])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()

    print("\nSimple OLS regression: avg_score ~ students_per_teacher")
    print(model_simple.summary())

    # Multiple regression with key controls
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    X_multi = sm.add_constant(df[["students_per_teacher"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()

    print(
        "\nMultiple OLS regression: "
        "avg_score ~ students_per_teacher + controls (CalWorks, lunch, expend, income, English learners)"
    )
    print(model_multi.summary())


if __name__ == "__main__":
    main()

