import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "hurricane.csv"

df = pd.read_csv(DATA_PATH)

# Core variables
fatalities = df["feature8"].astype(float)
log_fatalities = np.log1p(fatalities)

fem_index = df["feature4"].astype(float)
female_binary = df["feature6"].astype(float)

# Controls (severity and exposure proxies)
controls = pd.DataFrame({
    "category": df["feature7"].astype(float),
    "min_pressure": df["feature5"].astype(float),
    "max_wind": df["feature13"].astype(float),
    "log_damage": np.log1p(df["feature14"].astype(float)),
    "year": df["feature2"].astype(float),
})

# Build modeling frame
model_df = pd.concat([
    fatalities.rename("fatalities"),
    log_fatalities.rename("log_fatalities"),
    fem_index.rename("fem_index"),
    female_binary.rename("female"),
    controls,
], axis=1).dropna()

n = len(model_df)

# Simple correlations
corr_fem_logfatal = model_df[["fem_index", "log_fatalities"]].corr().iloc[0, 1]
corr_female_logfatal = model_df[["female", "log_fatalities"]].corr().iloc[0, 1]
spearman_fem_logfatal = model_df[["fem_index", "log_fatalities"]].corr(method="spearman").iloc[0, 1]
spearman_female_logfatal = model_df[["female", "log_fatalities"]].corr(method="spearman").iloc[0, 1]

# OLS with robust SEs
X_simple_fem = sm.add_constant(model_df[["fem_index"]])
ols_simple_fem = sm.OLS(model_df["log_fatalities"], X_simple_fem).fit(cov_type="HC3")

X_simple_female = sm.add_constant(model_df[["female"]])
ols_simple_female = sm.OLS(model_df["log_fatalities"], X_simple_female).fit(cov_type="HC3")

X_fem = sm.add_constant(model_df[["fem_index"] + list(controls.columns)])
ols_fem = sm.OLS(model_df["log_fatalities"], X_fem).fit(cov_type="HC3")

X_female = sm.add_constant(model_df[["female"] + list(controls.columns)])
ols_female = sm.OLS(model_df["log_fatalities"], X_female).fit(cov_type="HC3")

# Negative binomial GLM for counts
X_fem_nb = sm.add_constant(model_df[["fem_index"] + list(controls.columns)])
nb_fem = sm.GLM(model_df["fatalities"], X_fem_nb, family=sm.families.NegativeBinomial()).fit()

X_female_nb = sm.add_constant(model_df[["female"] + list(controls.columns)])
nb_female = sm.GLM(model_df["fatalities"], X_female_nb, family=sm.families.NegativeBinomial()).fit()

results = {
    "n": int(n),
    "correlations": {
        "fem_index_log_fatalities": float(corr_fem_logfatal),
        "female_log_fatalities": float(corr_female_logfatal),
        "spearman_fem_index_log_fatalities": float(spearman_fem_logfatal),
        "spearman_female_log_fatalities": float(spearman_female_logfatal),
    },
    "ols_simple_log_fatalities_fem_index": {
        "coef": float(ols_simple_fem.params["fem_index"]),
        "pvalue": float(ols_simple_fem.pvalues["fem_index"]),
    },
    "ols_simple_log_fatalities_female": {
        "coef": float(ols_simple_female.params["female"]),
        "pvalue": float(ols_simple_female.pvalues["female"]),
    },
    "ols_log_fatalities_fem_index": {
        "coef": float(ols_fem.params["fem_index"]),
        "pvalue": float(ols_fem.pvalues["fem_index"]),
    },
    "ols_log_fatalities_female": {
        "coef": float(ols_female.params["female"]),
        "pvalue": float(ols_female.pvalues["female"]),
    },
    "nb_fatalities_fem_index": {
        "coef": float(nb_fem.params["fem_index"]),
        "pvalue": float(nb_fem.pvalues["fem_index"]),
    },
    "nb_fatalities_female": {
        "coef": float(nb_female.params["female"]),
        "pvalue": float(nb_female.pvalues["female"]),
    },
}

print(json.dumps(results, indent=2))
