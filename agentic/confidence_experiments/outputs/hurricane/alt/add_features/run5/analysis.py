import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv("hurricane.csv")

# Basic prep
DF["log_deaths"] = np.log1p(DF["alldeaths"])
DF["log_ndam15"] = np.log1p(DF["ndam15"])

# Keep numeric severity controls
controls = ["wind", "min", "category", "year"]

# Drop rows with missing values in model vars
model_vars = ["log_deaths", "masfem", "gender_mf"] + controls
model_df = DF[model_vars].dropna().copy()

# Models
# 1) Simple association
m1 = smf.ols("log_deaths ~ masfem", data=model_df).fit(cov_type="HC3")
# 2) With controls for intensity and year
m2 = smf.ols("log_deaths ~ masfem + wind + min + category + year", data=model_df).fit(cov_type="HC3")
# 3) Binary gender with controls
m3 = smf.ols("log_deaths ~ gender_mf + wind + min + category + year", data=model_df).fit(cov_type="HC3")

# Also test damage as a proxy for caution (lower damage expected with more precautions)
# Use same controls for intensity
m4 = smf.ols("log_ndam15 ~ masfem + wind + min + category + year", data=DF.dropna(subset=["log_ndam15", "masfem"] + controls)).fit(cov_type="HC3")

summary = {
    "n": int(model_df.shape[0]),
    "m1_masfem_coef": m1.params.get("masfem"),
    "m1_masfem_p": m1.pvalues.get("masfem"),
    "m2_masfem_coef": m2.params.get("masfem"),
    "m2_masfem_p": m2.pvalues.get("masfem"),
    "m3_gender_mf_coef": m3.params.get("gender_mf"),
    "m3_gender_mf_p": m3.pvalues.get("gender_mf"),
    "m4_masfem_coef": m4.params.get("masfem"),
    "m4_masfem_p": m4.pvalues.get("masfem"),
}

print(json.dumps(summary, indent=2))
