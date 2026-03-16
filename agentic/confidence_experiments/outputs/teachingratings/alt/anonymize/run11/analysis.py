import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic clean: ensure proper types
# Identify columns by feature numbers per metadata
# feature6: beauty, feature7: teaching evaluation

# drop rows with missing key variables
key_cols = ['feature6', 'feature7']
df_clean = df.dropna(subset=key_cols).copy()

# Summary stats
summary = df_clean[key_cols].describe()

# Correlation
corr = df_clean['feature6'].corr(df_clean['feature7'])

# Bivariate OLS
model_simple = smf.ols('feature7 ~ feature6', data=df_clean).fit(cov_type='HC3')

# Multivariate OLS with controls
# Using available covariates per metadata: feature2,3,4,5,8,9,10,11,12
# Exclude feature1 (course id) and feature13 (instructor id)
formula = (
    'feature7 ~ feature6 + feature3 + C(feature4) + C(feature2) + C(feature5) + '
    'C(feature8) + C(feature9) + C(feature10) + feature11 + feature12'
)
model_controls = smf.ols(formula, data=df_clean).fit(cov_type='HC3')

# Standardized effect for beauty (simple and controls)
# Standardize beauty and outcome for comparable effect
zx = (df_clean['feature6'] - df_clean['feature6'].mean()) / df_clean['feature6'].std()
zy = (df_clean['feature7'] - df_clean['feature7'].mean()) / df_clean['feature7'].std()
std_df = df_clean.copy()
std_df['z_beauty'] = zx
std_df['z_eval'] = zy
std_simple = smf.ols('z_eval ~ z_beauty', data=std_df).fit(cov_type='HC3')

# Save results as json for summary
results = {
    'n': int(df_clean.shape[0]),
    'corr': float(corr),
    'simple_coef': float(model_simple.params['feature6']),
    'simple_p': float(model_simple.pvalues['feature6']),
    'simple_ci': [float(x) for x in model_simple.conf_int().loc['feature6'].tolist()],
    'controls_coef': float(model_controls.params['feature6']),
    'controls_p': float(model_controls.pvalues['feature6']),
    'controls_ci': [float(x) for x in model_controls.conf_int().loc['feature6'].tolist()],
    'std_beta': float(std_simple.params['z_beauty']),
    'std_p': float(std_simple.pvalues['z_beauty']),
}

print(json.dumps(results, indent=2))
