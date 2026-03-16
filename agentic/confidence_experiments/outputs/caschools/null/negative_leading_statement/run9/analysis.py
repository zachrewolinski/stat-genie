import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data = pd.read_csv("caschools.csv")
    data = data.copy()
    data["stratio"] = data["students"] / data["teachers"]
    data["testscr"] = (data["read"] + data["math"]) / 2.0

    variables = [
        "stratio",
        "testscr",
        "income",
        "lunch",
        "english",
        "computer",
        "expenditure",
    ]
    data_model = data[variables].dropna()

    correlation, correlation_p = stats.pearsonr(
        data_model["stratio"],
        data_model["testscr"],
    )

    outcome = data_model["testscr"]
    predictors_simple = sm.add_constant(data_model["stratio"])
    model_simple = sm.OLS(outcome, predictors_simple).fit()

    predictor_names = [
        "stratio",
        "income",
        "lunch",
        "english",
        "computer",
        "expenditure",
    ]
    predictors_full = sm.add_constant(data_model[predictor_names])
    model_full = sm.OLS(outcome, predictors_full).fit()

    coef_simple = model_simple.params["stratio"]
    pvalue_simple = model_simple.pvalues["stratio"]
    coef_full = model_full.params["stratio"]
    pvalue_full = model_full.pvalues["stratio"]

    predicted_change_five_students = -coef_full * 5.0

    print("Correlation between stratio and testscr:", correlation)
    print("Correlation p-value:", correlation_p)
    print("Simple OLS coefficient (stratio):", coef_simple)
    print("Simple OLS p-value (stratio):", pvalue_simple)
    print("Simple OLS R-squared:", model_simple.rsquared)
    print("Multiple OLS coefficient (stratio):", coef_full)
    print("Multiple OLS p-value (stratio):", pvalue_full)
    print("Multiple OLS R-squared:", model_full.rsquared)
    print(
        "Predicted test score change for 5 fewer",
        "students per teacher (full model):",
        predicted_change_five_students,
    )


if __name__ == "__main__":
    main()

