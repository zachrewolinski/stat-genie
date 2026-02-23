import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio (students per teacher)
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance measures
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Simple correlations
    for outcome in ["feature14", "feature15", "avg_score"]:
        r, p = stats.pearsonr(df["stratio"], df[outcome])
        print(f"Pearson correlation between stratio and {outcome}: r={r:.4f}, p={p:.4g}")

    # Simple linear regressions
    for outcome in ["feature14", "feature15", "avg_score"]:
        formula = f"{outcome} ~ stratio"
        model = smf.ols(formula=formula, data=df).fit()
        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        r2 = model.rsquared
        print(
            f"OLS {outcome} ~ stratio: coef={coef:.4f}, p={pval:.4g}, R^2={r2:.4f}"
        )

    # Multiple regressions with key controls
    controls = [
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]
    control_str = " + ".join(controls)

    for outcome in ["feature14", "feature15", "avg_score"]:
        formula = f"{outcome} ~ stratio + {control_str}"
        model = smf.ols(formula=formula, data=df).fit()
        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        r2 = model.rsquared
        print(
            f"Adj OLS {outcome} ~ stratio + controls: "
            f"coef={coef:.4f}, p={pval:.4g}, R^2={r2:.4f}"
        )


if __name__ == "__main__":
    main()

