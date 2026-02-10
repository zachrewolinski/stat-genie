import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # According to info.json descriptions:
    # english = total enrollment (students), students = number of teachers.
    df["studteach"] = df["english"] / df["students"]

    # Academic performance: average of reading and math scores.
    # district = avg reading, expenditure = avg math.
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    corr = df[["studteach", "testscr"]].corr().iloc[0, 1]

    X = sm.add_constant(df["studteach"])
    model = sm.OLS(df["testscr"], X).fit()

    print("Correlation between student-teacher ratio and test score:", corr)
    print("OLS slope (effect of +1 student per teacher):", model.params["studteach"])
    print("OLS t-stat for slope:", model.tvalues["studteach"])
    print("OLS p-value for slope:", model.pvalues["studteach"])


if __name__ == "__main__":
    main()

