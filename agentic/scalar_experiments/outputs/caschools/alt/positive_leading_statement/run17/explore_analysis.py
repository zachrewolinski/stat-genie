import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio (students per teacher).
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance as the average of reading and math scores.
    df["score"] = (df["read"] + df["math"]) / 2.0

    print("Summary of student-teacher ratio and scores:")
    print(df[["stratio", "score"]].describe())
    print()

    corr = df["stratio"].corr(df["score"])
    print(f"Correlation between student-teacher ratio and score: {corr:.3f}")
    print()

    # Simple bivariate regression.
    model_simple = smf.ols("score ~ stratio", data=df).fit()
    print("OLS: score ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for key demographics and resources.
    formula_controls = "score ~ stratio + income + english + lunch + calworks + expenditure + computer"
    model_controls = smf.ols(formula_controls, data=df).fit()
    print("OLS with controls:")
    print(model_controls.summary())
    print()

    # Mean scores by quartiles of student-teacher ratio (lower = smaller classes).
    df["stratio_q"] = pd.qcut(df["stratio"], 4, labels=False)
    group_means = df.groupby("stratio_q")["score"].mean()
    print("Mean score by student-teacher ratio quartile (0 = smallest classes):")
    print(group_means)


if __name__ == "__main__":
    main()

