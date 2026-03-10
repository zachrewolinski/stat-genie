import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Keep relevant columns for teaching ratings analysis
cols = ['eval','beauty','age','gender','minority','native','tenure','division','credits','students','allstudents']
missing_cols = [c for c in cols if c not in df.columns]

# Drop rows with missing values in key columns
use_cols = [c for c in cols if c in df.columns]
work = df[use_cols].copy()

# Basic summaries
summary = {
    'n_rows': len(df),
    'n_rows_used': len(work.dropna()),
    'missing_by_col': work.isna().sum().to_dict(),
    'beauty_mean': float(work['beauty'].mean()),
    'beauty_std': float(work['beauty'].std()),
    'eval_mean': float(work['eval'].mean()),
    'eval_std': float(work['eval'].std()),
}

# Simple linear regression eval ~ beauty
simple = smf.ols('eval ~ beauty', data=work).fit()

# Multiple regression with controls (categoricals as factors)
# Use students (participants) and exclude allstudents if present to reduce collinearity.
controls = ['age','gender','minority','native','tenure','division','credits','students']
controls = [c for c in controls if c in work.columns]
formula = 'eval ~ beauty'
if controls:
    formula += ' + ' + ' + '.join(controls)

multi = smf.ols(formula, data=work).fit()

# Pearson correlation
corr = work[['beauty','eval']].corr().iloc[0,1]

# Output key results
results = {
    'missing_cols': missing_cols,
    'summary': summary,
    'corr_beauty_eval': float(corr),
    'simple_coef': float(simple.params['beauty']),
    'simple_pvalue': float(simple.pvalues['beauty']),
    'simple_ci': [float(x) for x in simple.conf_int().loc['beauty'].tolist()],
    'simple_r2': float(simple.rsquared),
    'multi_formula': formula,
    'multi_coef': float(multi.params['beauty']),
    'multi_pvalue': float(multi.pvalues['beauty']),
    'multi_ci': [float(x) for x in multi.conf_int().loc['beauty'].tolist()],
    'multi_r2': float(multi.rsquared),
    'nobs_simple': int(simple.nobs),
    'nobs_multi': int(multi.nobs),
}

print(results)
