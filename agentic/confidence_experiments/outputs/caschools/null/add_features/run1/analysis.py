import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in key variables (should be none, but be safe)
    key_cols = ["stratio", "testscr", "income", "english", "lunch"]
    data = df[key_cols].dropna()

    # Correlation between student-teacher ratio and test scores
    corr, corr_p = stats.pearsonr(data["stratio"], data["testscr"])
    corr_read, corr_read_p = stats.pearsonr(data["stratio"], df.loc[data.index, "read"])
    corr_math, corr_math_p = stats.pearsonr(data["stratio"], df.loc[data.index, "math"])

    # Simple (bivariate) regression: testscr ~ stratio
    X_simple = sm.add_constant(data["stratio"])
    model_simple = sm.OLS(data["testscr"], X_simple).fit()

    # Multiple regression controlling for key demographics
    X_controls = data[["stratio", "income", "english", "lunch"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(data["testscr"], X_controls).fit()

    # Check robustness by trimming extreme student-teacher ratios (5th–95th percentile)
    q_low, q_high = data["stratio"].quantile([0.05, 0.95])
    trimmed = data[(data["stratio"] >= q_low) & (data["stratio"] <= q_high)]
    trim_corr, trim_corr_p = stats.pearsonr(trimmed["stratio"], trimmed["testscr"])
    X_trim = sm.add_constant(trimmed["stratio"])
    model_trim = sm.OLS(trimmed["testscr"], X_trim).fit()

    # Summarize key statistics we will use for the written conclusion
    summary = {
        "n_obs": int(len(data)),
        "corr_str_testscr": float(corr),
        "corr_p_value": float(corr_p),
        "corr_str_read": float(corr_read),
        "corr_str_read_p": float(corr_read_p),
        "corr_str_math": float(corr_math),
        "corr_str_math_p": float(corr_math_p),
        "mean_stratio": float(data["stratio"].mean()),
        "sd_stratio": float(data["stratio"].std()),
        "mean_testscr": float(data["testscr"].mean()),
        "sd_testscr": float(data["testscr"].std()),
        "simple_coef_str": float(model_simple.params["stratio"]),
        "simple_p_str": float(model_simple.pvalues["stratio"]),
        "simple_r2": float(model_simple.rsquared),
        "controls_coef_str": float(model_controls.params["stratio"]),
        "controls_p_str": float(model_controls.pvalues["stratio"]),
        "controls_r2": float(model_controls.rsquared),
        "trim_n_obs": int(len(trimmed)),
        "trim_q05_str": float(q_low),
        "trim_q95_str": float(q_high),
        "trim_corr_str_testscr": float(trim_corr),
        "trim_corr_p_value": float(trim_corr_p),
        "trim_simple_coef_str": float(model_trim.params["stratio"]),
        "trim_simple_p_str": float(model_trim.pvalues["stratio"]),
        "trim_simple_r2": float(model_trim.rsquared),
    }

    # Write a small machine-readable summary to inspect from the shell if needed
    # This file is only for intermediate inspection and is not the final output.
    pd.Series(summary).to_json("analysis_summary.json", indent=2)


if __name__ == "__main__":
    main()
