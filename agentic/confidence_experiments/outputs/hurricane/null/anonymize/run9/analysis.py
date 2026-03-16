import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [c for c in df.columns if c.startswith('feature') and c not in ['feature3', 'feature11']]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Define variables
fem_index = 'feature4'   # masculinity-femininity index (higher = more feminine)
binary_female = 'feature6'
fatalities = 'feature8'
pressure = 'feature5'    # min pressure
category = 'feature7'
wind = 'feature13'
year = 'feature2'

# Create log fatalities
# Add 1 to handle zero deaths
# Using natural log for interpretability

df['log_fatalities'] = np.log(df[fatalities] + 1)

# Drop rows with missing in variables used
base_vars = [fem_index, fatalities, pressure, category, wind, year]
model_df = df.dropna(subset=base_vars).copy()

# Model 1: simple association
m1 = smf.ols('log_fatalities ~ feature4', data=model_df).fit(cov_type='HC3')

# Model 2: controls for severity and year
m2 = smf.ols('log_fatalities ~ feature4 + feature5 + feature7 + feature13 + feature2', data=model_df).fit(cov_type='HC3')

# Model 3: interaction with severity (wind) to test stronger effect for severe storms
# Center variables to reduce multicollinearity
model_df['feature4_c'] = model_df[fem_index].mean()
model_df['feature13_c'] = model_df[wind].mean()
model_df['fem_c'] = model_df[fem_index] - model_df['feature4_c']
model_df['wind_c'] = model_df[wind] - model_df['feature13_c']

m3 = smf.ols('log_fatalities ~ fem_c + wind_c + fem_c:wind_c + feature5 + feature7 + feature2', data=model_df).fit(cov_type='HC3')

# Alternative: binary female name
m4 = smf.ols('log_fatalities ~ feature6 + feature5 + feature7 + feature13 + feature2', data=model_df).fit(cov_type='HC3')

# Spearman correlations for robustness
spearman_fem = model_df[[fem_index, fatalities]].corr(method='spearman').iloc[0,1]

# Summarize key results
results = {
    'n': len(model_df),
    'm1_coef': m1.params.get('feature4'),
    'm1_p': m1.pvalues.get('feature4'),
    'm2_coef': m2.params.get('feature4'),
    'm2_p': m2.pvalues.get('feature4'),
    'm3_interaction_coef': m3.params.get('fem_c:wind_c'),
    'm3_interaction_p': m3.pvalues.get('fem_c:wind_c'),
    'm3_fem_coef': m3.params.get('fem_c'),
    'm3_fem_p': m3.pvalues.get('fem_c'),
    'm4_coef': m4.params.get('feature6'),
    'm4_p': m4.pvalues.get('feature6'),
    'spearman_fem_fatal': spearman_fem,
}

print('Key results:')
for k, v in results.items():
    print(f"{k}: {v}")

# Save full model summaries to files for inspection
with open('model_summaries.txt', 'w') as f:
    f.write(m1.summary().as_text())
    f.write('\n\n')
    f.write(m2.summary().as_text())
    f.write('\n\n')
    f.write(m3.summary().as_text())
    f.write('\n\n')
    f.write(m4.summary().as_text())
