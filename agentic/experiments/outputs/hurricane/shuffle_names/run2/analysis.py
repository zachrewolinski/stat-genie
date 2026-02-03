import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv("hurricane.csv")

    # Columns are shuffled; rename based on metadata distributions and values
    df = df.rename(
        columns={
            "wind": "year",  # 1950-2012
            "alldeaths": "name",  # storm name (string)
            "category": "femininity",  # 1-11 femininity index
            "ndam15": "min_pressure",
            "masfem_mturk": "female_binary",  # 0/1
            "gender_mf": "ss_category",  # Saffir-Simpson category 1-5
            "name": "deaths",
            "elapsedyrs": "damage_2013",
            "masfem": "elapsed_years",
            "min": "source",
            "ind": "femininity_mturk",
            "year": "wind_speed",
            "source": "damage_2015",
        }
    )

    # Outcome: fatalities (skewed) -> log1p transform
    df["log_deaths"] = np.log1p(df["deaths"])

    # Core controls for storm severity and timing
    base_cols = ["femininity", "wind_speed", "min_pressure", "ss_category", "year"]
    base = df.dropna(subset=base_cols + ["log_deaths"])
    X_base = sm.add_constant(base[base_cols])
    model_base = sm.OLS(base["log_deaths"], X_base).fit()

    # Alternative model including damage (2013-adjusted) when available
    alt_cols = base_cols + ["damage_2013"]
    alt = df.dropna(subset=alt_cols + ["log_deaths"])
    X_alt = sm.add_constant(alt[alt_cols])
    model_alt = sm.OLS(alt["log_deaths"], X_alt).fit()

    # Binary female indicator model
    bin_cols = ["female_binary", "wind_speed", "min_pressure", "ss_category", "year"]
    bin_df = df.dropna(subset=bin_cols + ["log_deaths"])
    X_bin = sm.add_constant(bin_df[bin_cols])
    model_bin = sm.OLS(bin_df["log_deaths"], X_bin).fit()

    # Print key results for traceability
    print("Base model femininity coef:", model_base.params["femininity"], "p=", model_base.pvalues["femininity"])
    print("Alt model femininity coef:", model_alt.params["femininity"], "p=", model_alt.pvalues["femininity"])
    print("Binary female coef:", model_bin.params["female_binary"], "p=", model_bin.pvalues["female_binary"])


if __name__ == "__main__":
    main()
