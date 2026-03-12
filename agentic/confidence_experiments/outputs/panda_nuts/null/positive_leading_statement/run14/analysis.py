import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Standardize categorical values
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Compute efficiency rate: nuts opened per second
# Guard against zero seconds (shouldn't be)
df = df.copy()
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Poisson GLM with offset for time (seconds)
# Model nuts_opened counts with log(seconds) offset
# Predictors: age, sex, help
# Use cluster-robust SE by chimpanzee to account for repeated measures

# Encode categorical variables with treatment coding
# Build formula
formula = "nuts_opened ~ age + C(sex) + C(help)"

# Remove rows with missing values in relevant columns
model_df = df.dropna(subset=["nuts_opened", "seconds", "age", "sex", "help", "chimpanzee"]).copy()

# Add offset
offset = np.log(model_df["seconds"])

# Fit GLM Poisson
poisson_model = smf.glm(formula=formula, data=model_df,
                        family=sm.families.Poisson(), offset=offset).fit(cov_type='cluster', cov_kwds={'groups': model_df['chimpanzee']})

# Also fit OLS on log efficiency (log1p to handle zeros)
model_df["log_eff"] = np.log1p(model_df["efficiency"])
ols_model = smf.ols("log_eff ~ age + C(sex) + C(help)", data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['chimpanzee']})

# Summaries
print("Rows:", len(model_df), "Chimps:", model_df['chimpanzee'].nunique())
print("Efficiency summary:", model_df['efficiency'].describe())
print("\nPoisson GLM (cluster-robust):")
print(poisson_model.summary())
print("\nOLS log-eff (cluster-robust):")
print(ols_model.summary())

# Extract key results
results = {
    "poisson_params": poisson_model.params,
    "poisson_pvalues": poisson_model.pvalues,
    "poisson_ci": poisson_model.conf_int(),
    "ols_params": ols_model.params,
    "ols_pvalues": ols_model.pvalues,
    "ols_ci": ols_model.conf_int()
}

# Save key results
results_df = pd.DataFrame({
    "poisson_coef": poisson_model.params,
    "poisson_p": poisson_model.pvalues,
    "poisson_ci_low": poisson_model.conf_int()[0],
    "poisson_ci_high": poisson_model.conf_int()[1],
    "ols_coef": ols_model.params,
    "ols_p": ols_model.pvalues,
    "ols_ci_low": ols_model.conf_int()[0],
    "ols_ci_high": ols_model.conf_int()[1],
})

print("\nKey results:\n", results_df)

# Save for inspection
results_df.to_csv("analysis_results.csv")
