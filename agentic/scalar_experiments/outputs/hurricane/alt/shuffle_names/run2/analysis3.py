import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

path = "hurricane.csv"
df = pd.read_csv(path)

rename = {
    "ndam": "id",
    "wind": "year",
    "alldeaths": "name",
    "category": "femininity_index",
    "ndam15": "min_pressure",
    "masfem_mturk": "female_binary",
    "gender_mf": "ss_category",
    "name": "deaths",
    "elapsedyrs": "damage_2013",
    "masfem": "years_elapsed",
    "min": "source",
    "ind": "mturk_femininity",
    "year": "wind_speed",
    "source": "damage_2015",
}

df = df.rename(columns=rename)

# log transform deaths

df["log_deaths"] = np.log1p(df["deaths"].astype(float))

# standardize severity covariates
for col in ["wind_speed", "min_pressure", "ss_category"]:
    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

model_df = df.dropna(subset=["log_deaths", "wind_speed_z", "min_pressure_z", "ss_category_z"])

models = {}
formulas = {
    "femininity_index": "log_deaths ~ femininity_index + wind_speed_z + min_pressure_z + ss_category_z",
    "mturk_femininity": "log_deaths ~ mturk_femininity + wind_speed_z + min_pressure_z + ss_category_z",
    "female_binary": "log_deaths ~ female_binary + wind_speed_z + min_pressure_z + ss_category_z",
}

for name, formula in formulas.items():
    m = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    models[name] = m

for name, m in models.items():
    print("\nModel:", name)
    print(m.summary().tables[1])

