import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd


df = pd.read_csv('panda_nuts.csv')

# Clean variables
needed = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee']
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Normalize categories
sex_map = {'f': 1, 'm': 0, 'F': 1, 'M': 0}
help_map = {'y': 1, 'Y': 1, 'n': 0, 'N': 0, 'no': 0, 'yes': 1}

df = df.copy()

df['female_bin'] = df['sex'].map(sex_map)
if df['female_bin'].isna().any():
    # fall back to existing female column if present
    if 'female' in df.columns:
        df['female_bin'] = df['female']

df['help_bin'] = df['help'].map(help_map)

# Drop rows with missing key variables
analysis_df = df[['nuts_opened', 'seconds', 'age', 'female_bin', 'help_bin', 'chimpanzee']].dropna()

# Ensure positive seconds
analysis_df = analysis_df[analysis_df['seconds'] > 0].copy()
analysis_df['log_seconds'] = np.log(analysis_df['seconds'])

# Design matrix
X = analysis_df[['age', 'female_bin', 'help_bin']]
X = sm.add_constant(X)

y = analysis_df['nuts_opened']

# Poisson GLM with offset
# Poisson GLM with offset
model = sm.GLM(y, X, family=sm.families.Poisson(), offset=analysis_df['log_seconds'])
result = model.fit(cov_type='cluster', cov_kwds={'groups': analysis_df['chimpanzee']})

# Overdispersion check (Pearson chi2 / df)
pearson_chi2 = result.pearson_chi2
odf = result.df_resid
overdisp = pearson_chi2 / odf if odf > 0 else np.nan

# Compute rate ratios and CIs
params = result.params
bse = result.bse
rate_ratios = np.exp(params)
ci_lower = np.exp(params - 1.96 * bse)
ci_upper = np.exp(params + 1.96 * bse)

summary = {
    'n_rows': int(len(analysis_df)),
    'overdispersion': float(overdisp),
    'params': params.to_dict(),
    'pvalues': result.pvalues.to_dict(),
    'rate_ratios': rate_ratios.to_dict(),
    'ci_lower': ci_lower.to_dict(),
    'ci_upper': ci_upper.to_dict(),
}

# Negative binomial (handles overdispersion)
nb_model = smd.NegativeBinomial(y, X, offset=analysis_df['log_seconds'])
nb_res_robust = nb_model.fit(disp=False, cov_type='cluster', cov_kwds={'groups': analysis_df['chimpanzee']})

nb_params = nb_res_robust.params
nb_bse = nb_res_robust.bse
nb_pvalues = nb_res_robust.pvalues

nb_param_names = list(nb_res_robust.params.index)
alpha_name = 'alpha' if 'alpha' in nb_param_names else nb_param_names[-1]

coef_names = [name for name in nb_param_names if name != alpha_name]
nb_coef_params = nb_params[coef_names]
nb_coef_bse = nb_bse[coef_names]
nb_coef_pvalues = nb_pvalues[coef_names]

nb_rr = np.exp(nb_coef_params)
nb_ci_lower = np.exp(nb_coef_params - 1.96 * nb_coef_bse)
nb_ci_upper = np.exp(nb_coef_params + 1.96 * nb_coef_bse)

summary['negative_binomial'] = {
    'params': nb_coef_params.to_dict(),
    'pvalues': nb_coef_pvalues.to_dict(),
    'rate_ratios': nb_rr.to_dict(),
    'ci_lower': nb_ci_lower.to_dict(),
    'ci_upper': nb_ci_upper.to_dict(),
    'alpha': float(nb_params[alpha_name]),
}

print(json.dumps(summary, indent=2))
