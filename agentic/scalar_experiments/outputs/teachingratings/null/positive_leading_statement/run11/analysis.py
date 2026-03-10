import json
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic sanity: drop rows with missing values in key columns
key_cols = ['eval', 'beauty']

# Identify controls (categorical and numeric)
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
num_cols = ['age', 'students', 'allstudents']

# Keep only available columns (defensive)
cat_cols = [c for c in cat_cols if c in df.columns]
num_cols = [c for c in num_cols if c in df.columns]

# Build model formula
# Use eval as outcome, beauty as main predictor, with controls
rhs = ['beauty'] + num_cols + [f'C({c})' for c in cat_cols]
formula = 'eval ~ ' + ' + '.join(rhs)

# Drop rows with missing in any used columns
model_df = df[['eval','beauty'] + num_cols + cat_cols].dropna()

# Fit OLS with robust SE (HC3)
model = smf.ols(formula=formula, data=model_df).fit(cov_type='HC3')

# Also compute simple correlation
corr = model_df[['eval','beauty']].corr().iloc[0,1]

# Compute standardized effect (beta) for beauty: standardize variables and refit
z_df = model_df.copy()
for c in ['eval','beauty'] + num_cols:
    z_df[c] = (z_df[c] - z_df[c].mean()) / z_df[c].std(ddof=0)

z_formula = 'eval ~ ' + ' + '.join(['beauty'] + num_cols + [f'C({c})' for c in cat_cols])
z_model = smf.ols(formula=z_formula, data=z_df).fit(cov_type='HC3')

# Extract beauty stats
beauty_coef = model.params['beauty']
beauty_se = model.bse['beauty']
beauty_t = model.tvalues['beauty']
beauty_p = model.pvalues['beauty']

z_beauty_coef = z_model.params['beauty']

# Prepare summary stats
summary = {
    'n': int(model_df.shape[0]),
    'corr_eval_beauty': float(corr),
    'beauty_coef': float(beauty_coef),
    'beauty_se': float(beauty_se),
    'beauty_t': float(beauty_t),
    'beauty_p': float(beauty_p),
    'beauty_beta_standardized': float(z_beauty_coef),
    'r2': float(model.rsquared),
    'r2_adj': float(model.rsquared_adj),
    'formula': formula,
}

print(json.dumps(summary, indent=2))
