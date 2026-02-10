import pandas as pd
from scipy.stats import pearsonr


def main() -> None:
    df = pd.read_csv("caschools.csv")
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["academic_performance"] = df[["feature14", "feature15"]].mean(axis=1)

    subset = df[["student_teacher_ratio", "academic_performance"]].dropna()
    corr_avg, p_avg = pearsonr(
        subset["student_teacher_ratio"], subset["academic_performance"]
    )

    subset_read = df[["student_teacher_ratio", "feature14"]].dropna()
    corr_read, p_read = pearsonr(
        subset_read["student_teacher_ratio"], subset_read["feature14"]
    )

    subset_math = df[["student_teacher_ratio", "feature15"]].dropna()
    corr_math, p_math = pearsonr(
        subset_math["student_teacher_ratio"], subset_math["feature15"]
    )

    print("Correlation (avg score):", corr_avg, "P-value:", p_avg)
    print("Correlation (reading):  ", corr_read, "P-value:", p_read)
    print("Correlation (math):     ", corr_math, "P-value:", p_math)


if __name__ == "__main__":
    main()
