import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "teachingratings.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic correlation
    corr = df["beauty"].corr(df["eval"])
    n = df.shape[0]
    # Pearson correlation p-value
    r, p_corr = stats.pearsonr(df["beauty"], df["eval"])

    # Simple OLS
    model_simple = smf.ols("eval ~ beauty", data=df).fit()

    # Multivariate OLS with common controls
    # Use students (participated) as a proxy for evaluation reliability
    formula = (
        "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) "
        "+ C(division) + C(credits) + students + allstudents"
    )
    model_full = smf.ols(formula, data=df).fit()

    # Save results to json for inspection
    results = {
        "n": int(n),
        "corr": float(corr),
        "corr_p": float(p_corr),
        "simple_coef": float(model_simple.params["beauty"]),
        "simple_p": float(model_simple.pvalues["beauty"]),
        "simple_ci": [float(x) for x in model_simple.conf_int().loc["beauty"].tolist()],
        "full_coef": float(model_full.params["beauty"]),
        "full_p": float(model_full.pvalues["beauty"]),
        "full_ci": [float(x) for x in model_full.conf_int().loc["beauty"].tolist()],
        "simple_r2": float(model_simple.rsquared),
        "full_r2": float(model_full.rsquared),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
