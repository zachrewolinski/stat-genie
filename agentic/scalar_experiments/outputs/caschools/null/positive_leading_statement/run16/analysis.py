import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest
    vars_of_interest = ["stratio", "testscr", "calworks", "lunch", "english", "income"]
    df_model = df.dropna(subset=vars_of_interest).copy()

    # Simple Pearson correlation between student-teacher ratio and test scores
    r, p = stats.pearsonr(df_model["stratio"], df_model["testscr"])
    print("Correlation between STR (students per teacher) and testscr:")
    print(f"  r = {r:.3f}, p-value = {p:.3g}")

    # Linear regression controlling for demographics and income
    X = df_model[["stratio", "calworks", "lunch", "english", "income"]]
    X = sm.add_constant(X)
    y = df_model["testscr"]

    model = sm.OLS(y, X).fit()
    coef_str = model.params["stratio"]
    p_str = model.pvalues["stratio"]

    print("\nOLS regression of testscr on STR and controls:")
    print(f"  Coefficient on STR (students per teacher): {coef_str:.3f}")
    print(f"  p-value for STR: {p_str:.3g}")
    print(f"  R-squared: {model.rsquared:.3f}")

    # Difference in mean test scores between lowest and highest STR quartiles
    df_model["str_quartile"] = pd.qcut(df_model["stratio"], 4, labels=[1, 2, 3, 4])
    group_means = df_model.groupby("str_quartile")["testscr"].mean()
    print("\nAverage testscr by STR quartile (1=smallest classes, 4=largest):")
    for q, mean_val in group_means.items():
        print(f"  Quartile {int(q)}: {mean_val:.2f}")

    low = df_model[df_model["str_quartile"] == 1]["testscr"]
    high = df_model[df_model["str_quartile"] == 4]["testscr"]
    t_stat, p_t = stats.ttest_ind(low, high, equal_var=False)
    diff = low.mean() - high.mean()
    print(
        "\nDifference in mean testscr (Q1 - Q4 STR): "
        f"{diff:.2f}, t = {t_stat:.3f}, p-value = {p_t:.3g}"
    )


if __name__ == "__main__":
    main()
