import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: number of students per teacher
    df["str"] = df["students"] / df["teachers"]
    # Average test score across reading and math
    df["avg_score"] = (df["read"] + df["math"]) / 2

    # Keep variables needed for regression and drop missing values
    cols = ["avg_score", "str", "income", "english", "calworks", "lunch"]
    df_model = df[cols].dropna()

    # Simple bivariate regression: avg_score ~ str
    X_simple = sm.add_constant(df_model["str"])
    model_simple = sm.OLS(df_model["avg_score"], X_simple).fit()

    # Multiple regression with key socio-economic controls
    X_ctrl = sm.add_constant(df_model[["str", "income", "english", "calworks", "lunch"]])
    model_ctrl = sm.OLS(df_model["avg_score"], X_ctrl).fit()

    # Correlation
    corr = df_model["avg_score"].corr(df_model["str"])

    # Difference in means between low and high STR districts
    q25 = df_model["str"].quantile(0.25)
    q75 = df_model["str"].quantile(0.75)
    low_str = df_model[df_model["str"] <= q25]
    high_str = df_model[df_model["str"] >= q75]
    mean_low = low_str["avg_score"].mean()
    mean_high = high_str["avg_score"].mean()

    print("N_used:", len(df_model))
    print("Correlation_avgscore_str:", corr)

    print("Simple_OLS_coef_str:", model_simple.params["str"])
    print("Simple_OLS_pvalue_str:", model_simple.pvalues["str"])

    print("Ctrl_OLS_coef_str:", model_ctrl.params["str"])
    print("Ctrl_OLS_pvalue_str:", model_ctrl.pvalues["str"])

    print("Mean_avg_score:", df_model["avg_score"].mean())
    print("Mean_avg_score_low_STR_q25:", mean_low)
    print("Mean_avg_score_high_STR_q75:", mean_high)
    print("STR_q25:", q25)
    print("STR_q75:", q75)


if __name__ == "__main__":
    main()

