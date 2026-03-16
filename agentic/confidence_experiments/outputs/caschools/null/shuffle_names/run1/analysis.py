import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Reconstruct key constructs based on metadata mapping in info.json.
    # english: total enrollment; students: number of teachers.
    df["stratio"] = df["english"] / df["students"]

    # district: average reading score, expenditure: average math score.
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Basic sanity checks
    df = df.replace([pd.NA, float("inf"), -float("inf")], pd.NA).dropna(
        subset=["stratio", "testscr"]
    )

    # Simple bivariate regression: testscr on stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multivariate regression controlling for key covariates:
    # income (avg income), school (CalWorks %), computer (lunch %),
    # rownames (English learners %), grades (expenditure per student).
    covariates = ["stratio", "income", "school", "computer", "rownames", "grades"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["testscr"], X_multi).fit()

    # Extract key statistics
    simple_coef = model_simple.params["stratio"]
    simple_pval = model_simple.pvalues["stratio"]
    simple_r2 = model_simple.rsquared

    multi_coef = model_multi.params["stratio"]
    multi_pval = model_multi.pvalues["stratio"]
    multi_r2 = model_multi.rsquared

    summary = {
        "n_obs": int(model_multi.nobs),
        "simple_coef": float(simple_coef),
        "simple_pval": float(simple_pval),
        "simple_r2": float(simple_r2),
        "multi_coef": float(multi_coef),
        "multi_pval": float(multi_pval),
        "multi_r2": float(multi_r2),
    }

    out_path = Path("analysis_results.json")
    out_path.write_text(pd.Series(summary).to_json())


if __name__ == "__main__":
    main()

