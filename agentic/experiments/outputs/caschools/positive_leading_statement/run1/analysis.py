import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio
    df["str"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation
    corr = df[["str", "avg_score"]].corr().iloc[0, 1]

    # Simple OLS: avg_score ~ str
    X_simple = sm.add_constant(df[["str"]])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit(cov_type="HC1")

    # Multivariate OLS with common controls
    controls = ["lunch", "english", "income", "expenditure"]
    X_ctrl = sm.add_constant(df[["str"] + controls])
    model_ctrl = sm.OLS(df["avg_score"], X_ctrl).fit(cov_type="HC1")

    print("Rows:", len(df))
    print("Correlation(str, avg_score):", corr)
    print("\nSimple OLS: avg_score ~ str (HC1 robust)")
    print(model_simple.summary().tables[1])
    print("\nControlled OLS: avg_score ~ str + lunch + english + income + expenditure (HC1 robust)")
    print(model_ctrl.summary().tables[1])

    # Save key results for easier reuse if needed
    results = {
        "corr": corr,
        "simple_coef": model_simple.params["str"],
        "simple_p": model_simple.pvalues["str"],
        "ctrl_coef": model_ctrl.params["str"],
        "ctrl_p": model_ctrl.pvalues["str"],
    }
    pd.Series(results).to_csv("analysis_results.csv", header=False)


if __name__ == "__main__":
    main()
