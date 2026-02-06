import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("hurricane.csv")

    # Basic derived variables
    df["log_deaths"] = np.log1p(df["alldeaths"])
    for col in ["min", "wind", "category"]:
        df[f"{col}_c"] = df[col] - df[col].mean()

    # Descriptives
    desc = {
        "n": len(df),
        "mean_deaths": df["alldeaths"].mean(),
        "mean_log_deaths": df["log_deaths"].mean(),
        "mean_masfem": df["masfem"].mean(),
        "corr_masfem_log_deaths": df["masfem"].corr(df["log_deaths"]),
    }
    print("DESCRIPTIVES")
    for k, v in desc.items():
        print(f"{k}: {v}")

    # Group comparison by binary gender
    group_means = df.groupby("gender_mf")["alldeaths"].mean()
    print("\nMEAN DEATHS BY GENDER_MF (0=male,1=female)")
    print(group_means)

    # Main regression: log deaths on name femininity, controlling for storm intensity
    m1 = smf.ols("log_deaths ~ masfem + wind + min + category", data=df).fit()
    print("\nOLS: log_deaths ~ masfem + wind + min + category")
    print(m1.summary().tables[1])

    # Interaction check (as in some prior analyses): femininity x minimum pressure
    m2 = smf.ols("log_deaths ~ masfem * min_c + wind_c + category_c", data=df).fit()
    print("\nOLS: log_deaths ~ masfem * min_c + wind_c + category_c")
    print(m2.summary().tables[1])

    # Store key results for quick reference
    key = pd.DataFrame(
        {
            "model": ["m1", "m2"],
            "coef_masfem": [m1.params["masfem"], m2.params["masfem"]],
            "p_masfem": [m1.pvalues["masfem"], m2.pvalues["masfem"]],
            "coef_masfem_x_min_c": [np.nan, m2.params.get("masfem:min_c", np.nan)],
            "p_masfem_x_min_c": [np.nan, m2.pvalues.get("masfem:min_c", np.nan)],
        }
    )
    print("\nKEY RESULTS")
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
