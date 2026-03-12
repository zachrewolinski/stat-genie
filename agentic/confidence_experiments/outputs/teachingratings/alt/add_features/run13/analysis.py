import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('teachingratings.csv')

# Ensure columns
print('columns', df.columns.tolist())

# Basic stats

# Drop rows with missing beauty or eval
sub = df[['beauty','eval']].dropna()
print('n', len(sub))

# Correlation
corr = sub['beauty'].corr(sub['eval'])
print('corr', corr)

# Simple regression
model1 = smf.ols('eval ~ beauty', data=df).fit()
print(model1.summary())

# Multiple regression with available controls (subset that are in dataset)
# Identify potential controls from known set
controls = [c for c in ['age','gender','minority','credits','division','native','tenure','students','allstudents','prof'] if c in df.columns]

# For categorical controls, statsmodels formula will handle if they are object/category
if controls:
    formula = 'eval ~ beauty + ' + ' + '.join(controls)
    model2 = smf.ols(formula, data=df).fit()
    print('formula', formula)
    print(model2.summary())
else:
    model2 = None

# Save key results
results = {
    'n': int(len(sub)),
    'corr': corr,
    'model1': {
        'coef_beauty': model1.params.get('beauty', np.nan),
        'p_beauty': model1.pvalues.get('beauty', np.nan),
        'r2': model1.rsquared,
    },
    'model2': None
}
if model2 is not None:
    results['model2'] = {
        'coef_beauty': model2.params.get('beauty', np.nan),
        'p_beauty': model2.pvalues.get('beauty', np.nan),
        'r2': model2.rsquared,
        'nobs': int(model2.nobs)
    }

with open('analysis_results.json','w') as f:
    json.dump(results,f,indent=2)

print('saved analysis_results.json')
