import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and overall academic performance.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    read_score = df["feature14"]
    math_score = df["feature15"]

    ratio = df["student_teacher_ratio"]
    avg_score = df["avg_score"]

    corr_avg = ratio.corr(avg_score)
    corr_read = ratio.corr(read_score)
    corr_math = ratio.corr(math_score)

    # Simple bivariate OLS: avg_score ~ student_teacher_ratio
    X = sm.add_constant(ratio)
    model = sm.OLS(avg_score, X).fit()

    print("Number of districts:", len(df))
    print("Correlation (ratio, avg_score):", corr_avg)
    print("Correlation (ratio, reading):", corr_read)
    print("Correlation (ratio, math):", corr_math)
    print("OLS slope (avg_score ~ ratio):", model.params["student_teacher_ratio"])
    print("t-value for slope:", model.tvalues["student_teacher_ratio"])
    print("p-value for slope:", model.pvalues["student_teacher_ratio"])
    print("R-squared:", model.rsquared)


if __name__ == "__main__":
    main()
