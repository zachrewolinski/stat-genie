import json

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2

    df = df.dropna(subset=["stratio", "testscr"])

    # Simple bivariate relationship
    corr = df["stratio"].corr(df["testscr"])

    # Linear regressions: simple and with basic controls
    m_simple = smf.ols("testscr ~ stratio", data=df).fit()
    m_adj = smf.ols(
        "testscr ~ stratio + income + english + lunch + calworks + expenditure",
        data=df,
    ).fit()

    results = {
        "corr_stratio_testscr": float(corr),
        "simple_coef_stratio": float(m_simple.params["stratio"]),
        "simple_p_stratio": float(m_simple.pvalues["stratio"]),
        "simple_r2": float(m_simple.rsquared),
        "adj_coef_stratio": float(m_adj.params["stratio"]),
        "adj_p_stratio": float(m_adj.pvalues["stratio"]),
        "adj_r2": float(m_adj.rsquared),
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

