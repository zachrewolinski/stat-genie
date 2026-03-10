import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Rename for readability
rename_map = {
    "feature1": "id",
    "feature2": "year",
    "feature3": "name",
    "feature4": "fem_index",
    "feature5": "min_pressure",
    "feature6": "female_name",
    "feature7": "category",
    "feature8": "deaths",
    "feature9": "damage_2013",
    "feature10": "years_since",
    "feature11": "source",
    "feature12": "mturk_fem",
    "feature13": "wind_speed",
    "feature14": "damage_2015",
}

df = df.rename(columns=rename_map)

# Basic cleaning
numeric_cols = [
    "year",
    "fem_index",
    "min_pressure",
    "female_name",
    "category",
    "deaths",
    "damage_2013",
    "years_since",
    "mturk_fem",
    "wind_speed",
    "damage_2015",
]

# Ensure numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Outcome transforms
# deaths and damages are skewed; log1p to handle zeros
for col in ["deaths", "damage_2013", "damage_2015"]:
    df[f"log1p_{col}"] = np.log1p(df[col])

# Severity index: higher = stronger
severity_components = pd.DataFrame(
    {
        "wind_speed": df["wind_speed"],
        "category": df["category"],
        "pressure_neg": -df["min_pressure"],
    }
)
severity_z = (severity_components - severity_components.mean()) / severity_components.std(ddof=0)

df["severity_z"] = severity_z.mean(axis=1)

# Models
models = {}

# Simple association
models["m1_simple"] = smf.ols("log1p_deaths ~ fem_index", data=df).fit(cov_type="HC3")

# Control for severity and year
models["m2_controls"] = smf.ols(
    "log1p_deaths ~ fem_index + wind_speed + min_pressure + category + year",
    data=df,
).fit(cov_type="HC3")

# Interaction with severity
models["m3_interaction"] = smf.ols(
    "log1p_deaths ~ fem_index * severity_z + year",
    data=df,
).fit(cov_type="HC3")

# Binary female indicator
models["m4_binary"] = smf.ols(
    "log1p_deaths ~ female_name + wind_speed + min_pressure + category + year",
    data=df,
).fit(cov_type="HC3")

# Damage outcome as alternative proxy (2013 adjusted)
models["m5_damage"] = smf.ols(
    "log1p_damage_2013 ~ fem_index + wind_speed + min_pressure + category + year",
    data=df,
).fit(cov_type="HC3")


def summarize_model(model):
    params = model.params
    pvals = model.pvalues
    conf = model.conf_int(alpha=0.05)
    return {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "coef_fem": float(params.get("fem_index", np.nan)),
        "p_fem": float(pvals.get("fem_index", np.nan)),
        "ci_fem_low": float(conf.loc["fem_index", 0]) if "fem_index" in conf.index else np.nan,
        "ci_fem_high": float(conf.loc["fem_index", 1]) if "fem_index" in conf.index else np.nan,
        "coef_female": float(params.get("female_name", np.nan)),
        "p_female": float(pvals.get("female_name", np.nan)),
        "coef_interaction": float(params.get("fem_index:severity_z", np.nan)),
        "p_interaction": float(pvals.get("fem_index:severity_z", np.nan)),
    }

summary = {name: summarize_model(model) for name, model in models.items()}

# Correlations for context
corr_fem_deaths = df[["fem_index", "deaths"]].corr().iloc[0, 1]

output = {
    "summary": summary,
    "corr_fem_deaths": float(corr_fem_deaths),
    "deaths_stats": {
        "min": float(df["deaths"].min()),
        "max": float(df["deaths"].max()),
        "median": float(df["deaths"].median()),
        "mean": float(df["deaths"].mean()),
    },
}

print(json.dumps(output, indent=2))
