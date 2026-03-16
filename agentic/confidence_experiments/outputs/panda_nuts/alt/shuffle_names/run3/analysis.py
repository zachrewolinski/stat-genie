import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Load data

df = pd.read_csv('panda_nuts.csv')

# Map columns based on observed values and metadata mismatch
# age: numeric 1-22
# sex: in 'nuts_opened' column (m/f)
# help: in 'seconds' column (y/N)
# nuts opened count: in 'help' column
# session duration seconds: in 'chimpanzee' column

# Clean and compute efficiency

df = df.copy()

# Ensure proper types

df['age_years'] = pd.to_numeric(df['age'], errors='coerce')
df['sex_mf'] = df['nuts_opened'].astype(str).str.strip()
df['help_yn'] = df['seconds'].astype(str).str.strip()

df['nuts_opened_count'] = pd.to_numeric(df['help'], errors='coerce')
df['duration_seconds'] = pd.to_numeric(df['chimpanzee'], errors='coerce')

# Efficiency: nuts opened per second
# Avoid division by zero

df['efficiency'] = df['nuts_opened_count'] / df['duration_seconds']

# Drop rows with missing required fields

model_df = df.dropna(subset=['age_years', 'sex_mf', 'help_yn', 'efficiency'])

# Encode categorical variables

model_df['sex_mf'] = model_df['sex_mf'].astype('category')
model_df['help_yn'] = model_df['help_yn'].astype('category')

# Fit OLS model

model = smf.ols('efficiency ~ age_years + C(sex_mf) + C(help_yn)', data=model_df).fit()

# Also fit model with log efficiency to handle skewness (add small constant)

model_df['log_eff'] = np.log(model_df['efficiency'] + 1e-6)
model_log = smf.ols('log_eff ~ age_years + C(sex_mf) + C(help_yn)', data=model_df).fit()

# ANOVA for overall effects

anova = anova_lm(model, typ=2)
anova_log = anova_lm(model_log, typ=2)

# Summaries for output

summary = {
    'n_rows': int(len(model_df)),
    'efficiency_mean': float(model_df['efficiency'].mean()),
    'efficiency_median': float(model_df['efficiency'].median()),
    'efficiency_std': float(model_df['efficiency'].std()),
    'model_params': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_r2': float(model.rsquared),
    'anova': anova.reset_index().to_dict(orient='records'),
    'model_log_params': model_log.params.to_dict(),
    'model_log_pvalues': model_log.pvalues.to_dict(),
    'model_log_r2': float(model_log.rsquared),
    'anova_log': anova_log.reset_index().to_dict(orient='records'),
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Analysis complete. Summary written to analysis_summary.json')
