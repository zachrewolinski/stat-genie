import json
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Ensure columns of interest exist
cols = ['eval','beauty','age','gender','minority','credits','division','native','tenure','students','allstudents']
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing values in relevant columns
sub = df[cols].dropna()

# Basic correlation
pearson_r, pearson_p = stats.pearsonr(sub['beauty'], sub['eval'])

# Simple regression
model_simple = smf.ols('eval ~ beauty', data=sub).fit()

# Multiple regression with covariates
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents'
model_full = smf.ols(formula, data=sub).fit()

result = {
    'n': int(sub.shape[0]),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'simple_coef_beauty': float(model_simple.params.get('beauty', float('nan'))),
    'simple_p_beauty': float(model_simple.pvalues.get('beauty', float('nan'))),
    'simple_r2': float(model_simple.rsquared),
    'full_coef_beauty': float(model_full.params.get('beauty', float('nan'))),
    'full_p_beauty': float(model_full.pvalues.get('beauty', float('nan'))),
    'full_r2': float(model_full.rsquared),
}

print(json.dumps(result, indent=2))
