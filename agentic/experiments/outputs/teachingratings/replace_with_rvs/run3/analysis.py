import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic cleaning: ensure categorical variables treated as category
cat_cols = ['minority','gender','credits','division','native','tenure']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# Main model: eval on beauty with controls
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Also run a simple bivariate model for reference
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Save key results for inspection
results = {
    'n': int(model.nobs),
    'coef_beauty': float(model.params['beauty']),
    'p_beauty': float(model.pvalues['beauty']),
    'coef_beauty_simple': float(model_simple.params['beauty']),
    'p_beauty_simple': float(model_simple.pvalues['beauty']),
    'r2': float(model.rsquared)
}

# Write a brief text summary for human check
with open('analysis_summary.txt','w') as f:
    f.write(model.summary().as_text())
    f.write('\n\nSimple model:\n')
    f.write(model_simple.summary().as_text())
    f.write('\n\nKey results:\n')
    for k,v in results.items():
        f.write(f'{k}: {v}\n')
