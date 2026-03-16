import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and overall test score
    df["str"] = df["feature6"] / df["feature7"]
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    df = df.dropna(subset=["str", "testscr"])

    print(f"N (districts) = {len(df)}")

    # Descriptive statistics
    print("\nStudent-teacher ratio (str) summary:")
    print(df["str"].describe())

    print("\nAcademic performance (testscr = mean of reading and math) summary:")
    print(df["testscr"].describe())

    # Bivariate Pearson correlation
    r, p = stats.pearsonr(df["str"], df["testscr"])
    print(f"\nPearson correlation between str and testscr: r = {r:.3f}, p-value = {p:.3g}")

    # Simple linear regression: testscr ~ str
    X = sm.add_constant(df["str"])
    y = df["testscr"]
    model_simple = sm.OLS(y, X).fit()

    print("\nOLS regression: testscr ~ str")
    print(model_simple.summary())

    # Add a simple set of controls capturing demographics and resources
    df["comp_stu"] = df["feature10"] / df["feature6"]
    controls = ["str", "feature8", "feature9", "feature11", "feature12", "feature13", "comp_stu"]
    Xc = sm.add_constant(df[controls])
    model_controls = sm.OLS(y, Xc).fit()

    print("\nOLS regression with controls:")
    print("testscr ~ str + CalWorks% + lunch% + expenditure + income + English% + computers per student")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

