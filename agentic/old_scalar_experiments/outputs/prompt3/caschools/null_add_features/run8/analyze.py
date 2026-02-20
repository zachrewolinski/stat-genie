import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables (should be none, but safe)
    df_model = df.dropna(subset=["stratio", "score"])

    # Simple bivariate regression: average score on student-teacher ratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["score"], X_simple).fit()

    # Multiple regression controlling for important covariates
    covariates = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    covariates = [c for c in covariates if c in df_model.columns]
    X_controls = sm.add_constant(df_model[["stratio"] + covariates])
    model_controls = sm.OLS(df_model["score"], X_controls).fit()

    # Collect key results
    simple_coef = model_simple.params["stratio"]
    simple_pval = model_simple.pvalues["stratio"]
    simple_r2 = model_simple.rsquared

    controls_coef = model_controls.params["stratio"]
    controls_pval = model_controls.pvalues["stratio"]
    controls_r2 = model_controls.rsquared

    # Save a compact summary to inspect manually if needed
    with open("analysis_summary.txt", "w") as f:
        f.write("Simple model: score ~ stratio\n")
        f.write(model_simple.summary().as_text())
        f.write("\n\nControls model: score ~ stratio + covariates\n")
        f.write(model_controls.summary().as_text())
        f.write("\n\nKey coefficients:\n")
        f.write(
            f"simple_coef={simple_coef:.4f}, p={simple_pval:.4g}, R2={simple_r2:.3f}\n"
        )
        f.write(
            f"controls_coef={controls_coef:.4f}, p={controls_pval:.4g}, R2={controls_r2:.3f}\n"
        )


if __name__ == "__main__":
    main()

