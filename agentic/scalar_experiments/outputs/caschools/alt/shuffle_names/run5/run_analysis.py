import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to info.json, column names are shuffled relative to their meanings.
    # Map to interpretable variables:
    # - "english" column: total enrollment (number of students)
    # - "students" column: number of teachers
    # - "district" column: average reading score
    # - "expenditure" column: average math score
    df["students_total"] = df["english"]
    df["teachers_n"] = df["students"]

    # Student-teacher ratio and academic performance
    df["stratio"] = df["students_total"] / df["teachers_n"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Covariates from metadata:
    # - "income": district average income (in $1,000s)
    # - "school": percent qualifying for CalWorks (income assistance)
    # - "rownames": percent of English learners
    model_cols = ["testscr", "stratio", "income", "school", "rownames"]
    df_model = df[model_cols].dropna().copy()

    # Correlation
    corr = df_model["testscr"].corr(df_model["stratio"])
    print("Correlation between testscr and stratio:", corr)

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with key demographic controls
    X_multi = sm.add_constant(df_model[["stratio", "income", "school", "rownames"]])
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()
    print("\nMultiple OLS: testscr ~ stratio + income + school + rownames")
    print(model_multi.summary())

    # Key statistics on the student-teacher ratio effect
    coef = model_multi.params["stratio"]
    pval = model_multi.pvalues["stratio"]
    beta = coef * df_model["stratio"].std() / df_model["testscr"].std()

    stratio_range = np.percentile(df_model["stratio"], 90) - np.percentile(
        df_model["stratio"], 10
    )
    score_change = coef * stratio_range

    print("\nCoefficient (multiple regression) on stratio:", coef)
    print("p-value:", pval)
    print("Standardized effect (beta):", beta)
    print(
        "Approx. test score change from 10th to 90th percentile stratio:",
        score_change,
    )


if __name__ == "__main__":
    main()

