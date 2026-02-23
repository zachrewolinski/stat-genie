import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meanings (based on info.json)
    df = df.copy()
    df["enrollment"] = df["english"]  # Total enrollment
    df["teachers_n"] = df["students"]  # Number of teachers

    # Student–teacher ratio: students per teacher
    df = df[df["teachers_n"] > 0].copy()
    df["stratio"] = df["enrollment"] / df["teachers_n"]

    # Academic performance: reading, math, and their average
    df["read_scr"] = df["district"]  # Average reading score
    df["math_scr"] = df["expenditure"]  # Average math score
    df["testscr"] = df[["read_scr", "math_scr"]].mean(axis=1)

    # Socioeconomic and resource controls
    df["calworks_pct"] = df["school"]  # Percent on CalWorks
    df["lunch_pct"] = df["computer"]  # Percent on reduced-price lunch
    df["expn_stu"] = df["grades"]  # Expenditure per student
    df["avginc"] = df["income"]  # District average income (in $1,000)
    df["el_pct"] = df["rownames"]  # Percent English learners

    df = df.dropna(
        subset=[
            "stratio",
            "testscr",
            "calworks_pct",
            "lunch_pct",
            "expn_stu",
            "avginc",
            "el_pct",
        ]
    )

    print("Number of observations:", len(df))
    print("\nStudent–teacher ratio (stratio) summary:")
    print(df["stratio"].describe())

    print("\nAverage test score (testscr) summary:")
    print(df["testscr"].describe())

    corr = df["testscr"].corr(df["stratio"])
    print("\nCorrelation between testscr and stratio:", corr)

    # Simple bivariate regression
    ols1 = smf.ols("testscr ~ stratio", data=df).fit()
    print("\nOLS1: testscr ~ stratio")
    print(ols1.summary())

    # Multiple regression with key controls
    ols2 = smf.ols(
        "testscr ~ stratio + calworks_pct + lunch_pct + el_pct + avginc + expn_stu",
        data=df,
    ).fit()
    print("\nOLS2: testscr ~ stratio + controls")
    print(ols2.summary())


if __name__ == "__main__":
    main()

