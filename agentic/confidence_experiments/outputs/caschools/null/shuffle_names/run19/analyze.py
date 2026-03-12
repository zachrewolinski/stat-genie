import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Map shuffled columns to conceptual variables based on metadata
    # Total enrollment
    enroll = df["english"]
    # Number of teachers (FTE)
    teachers = df["students"]
    # Student–teacher ratio: students per teacher
    stratio = enroll / teachers

    # Academic performance: use both reading and math averages
    read_score = df["district"]
    math_score = df["expenditure"]
    avg_score = (read_score + math_score) / 2.0

    # Basic summaries
    print("Student–teacher ratio summary:")
    print(stratio.describe())
    print()

    corr_read = read_score.corr(stratio)
    corr_math = math_score.corr(stratio)
    corr_avg = avg_score.corr(stratio)

    print("Correlation (ratio vs. reading):", corr_read)
    print("Correlation (ratio vs. math):   ", corr_math)
    print("Correlation (ratio vs. average):", corr_avg)
    print()

    # Linear regression: average score on student–teacher ratio
    X = sm.add_constant(stratio)
    model_simple = sm.OLS(avg_score, X).fit()

    print("=== Simple OLS: avg_score ~ stratio (full sample) ===")
    print(model_simple.summary())
    print()

    # Add common controls: income, expenditure per student, poverty/ell proxies
    controls = df[["income", "grades", "school", "computer", "rownames"]]
    Xc = sm.add_constant(pd.concat([stratio, controls], axis=1))
    model_controls = sm.OLS(avg_score, Xc).fit()

    print("=== OLS with controls: avg_score ~ stratio + controls (full sample) ===")
    print(model_controls.summary())
    print()

    # Trimmed sample: focus on more plausible class sizes
    trimmed_mask = (stratio >= 5) & (stratio <= 40)
    df_trim = df.loc[trimmed_mask].copy()
    stratio_t = (df_trim["english"] / df_trim["students"]).astype(float)
    read_t = df_trim["district"]
    math_t = df_trim["expenditure"]
    avg_t = (read_t + math_t) / 2.0

    print("Number of districts in trimmed sample:", df_trim.shape[0])
    print("Correlation (trimmed ratio vs. average score):", avg_t.corr(stratio_t))
    print()

    Xt = sm.add_constant(stratio_t)
    model_trim = sm.OLS(avg_t, Xt).fit()
    print("=== Simple OLS (trimmed): avg_score ~ stratio ===")
    print(model_trim.summary())


if __name__ == "__main__":
    main()
