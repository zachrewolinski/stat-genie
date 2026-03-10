import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def ols_with_controls(df: pd.DataFrame, y: str, x: str, controls: list[str]):
    cols = [y, x] + controls
    data = df[cols].dropna()
    yv = data[y]
    X = data[[x] + controls]
    X = sm.add_constant(X)
    model = sm.OLS(yv, X).fit(cov_type="HC3")
    return model, data.shape[0]


def summarize_model(model: sm.regression.linear_model.RegressionResultsWrapper, x: str):
    coef = model.params.get(x, np.nan)
    pval = model.pvalues.get(x, np.nan)
    return coef, pval


def main():
    df = load_data("hurricane.csv")

    # Outcome variables
    df = df.copy()
    df["log_deaths"] = np.log1p(df["alldeaths"])
    df["log_ndam"] = np.log(df["ndam"].replace(0, np.nan))
    df["log_ndam15"] = np.log(df["ndam15"].replace(0, np.nan))

    results = []

    # Simple bivariate
    model, n = ols_with_controls(df, "log_deaths", "masfem", [])
    coef, pval = summarize_model(model, "masfem")
    results.append(("bivariate_log_deaths", n, coef, pval))

    # Controls for storm intensity and exposure proxies
    controls = ["wind", "min", "category", "log_ndam", "year"]
    model, n = ols_with_controls(df, "log_deaths", "masfem", controls)
    coef, pval = summarize_model(model, "masfem")
    results.append(("controls_log_deaths", n, coef, pval))

    # Alternative damage normalization
    controls = ["wind", "min", "category", "log_ndam15", "year"]
    model, n = ols_with_controls(df, "log_deaths", "masfem", controls)
    coef, pval = summarize_model(model, "masfem")
    results.append(("controls_log_deaths_ndam15", n, coef, pval))

    # Binary gender indicator
    controls = ["wind", "min", "category", "log_ndam", "year"]
    model, n = ols_with_controls(df, "log_deaths", "gender_mf", controls)
    coef, pval = summarize_model(model, "gender_mf")
    results.append(("controls_log_deaths_gender", n, coef, pval))

    # Store key results for later interpretation
    out = []
    for name, n, coef, pval in results:
        out.append({
            "model": name,
            "n": int(n),
            "coef": float(coef),
            "pval": float(pval),
        })

    with open("analysis_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
