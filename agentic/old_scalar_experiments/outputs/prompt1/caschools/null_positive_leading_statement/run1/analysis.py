import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Students per teacher: lower values mean smaller classes.
    df["str"] = df["students"] / df["teachers"]
    df["score"] = df[["read", "math"]].mean(axis=1)

    corr = df["str"].corr(df["score"])

    model = sm.OLS(df["score"], sm.add_constant(df["str"])).fit()

    controls = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    X_controls = sm.add_constant(df[["str"] + controls])
    model_controls = sm.OLS(df["score"], X_controls).fit()

    # Quartile comparison for additional intuition.
    df["str_quartile"] = pd.qcut(df["str"], 4, labels=False)
    low_q = df[df["str_quartile"] == 0]["score"].mean()
    high_q = df[df["str_quartile"] == 3]["score"].mean()

    print("Number of districts:", len(df))
    print("Correlation between students-per-teacher and score:", corr)
    print("OLS slope for score ~ str:", model.params["str"])
    print("p-value for slope:", model.pvalues["str"])
    print(
        "OLS slope for score ~ str + controls (income, english, lunch, calworks, computer, expenditure):",
        model_controls.params["str"],
    )
    print("p-value for str in model with controls:", model_controls.pvalues["str"])
    print("Mean score, lowest class-size quartile:", low_q)
    print("Mean score, highest class-size quartile:", high_q)


if __name__ == "__main__":
    main()
