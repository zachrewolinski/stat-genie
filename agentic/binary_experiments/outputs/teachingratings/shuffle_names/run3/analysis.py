import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic sanity: drop obvious identifier-like column
# 'division' is unique 1..463 in this shuffled dataset; treat as row id.
df = _df.copy()
if df['division'].nunique() == len(df):
    df = df.drop(columns=['division'])

# Ensure categorical columns are treated as category
cat_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# Outcome: student instructional rating (1-5)
# In this dataset, 'allstudents' matches the evaluation scale.

# Model 1: bivariate
model1 = smf.ols('allstudents ~ beauty', data=df).fit()

# Model 2: add available controls (excluding id-like division)
# Use numeric and categorical covariates except outcome and beauty
controls = []
for c in df.columns:
    if c in ['allstudents', 'beauty']:
        continue
    # exclude any remaining identifier-like column
    if c == 'division':
        continue
    controls.append(c)

formula2 = 'allstudents ~ beauty'
if controls:
    formula2 += ' + ' + ' + '.join(controls)
model2 = smf.ols(formula2, data=df).fit()

# Save key results for reporting
results = {
    'n': int(df.shape[0]),
    'model1_coef': float(model1.params['beauty']),
    'model1_pval': float(model1.pvalues['beauty']),
    'model2_coef': float(model2.params['beauty']),
    'model2_pval': float(model2.pvalues['beauty']),
}

print('N', results['n'])
print('Model1 beauty coef', results['model1_coef'], 'p', results['model1_pval'])
print('Model2 beauty coef', results['model2_coef'], 'p', results['model2_pval'])

# Also compute standardized effect for context
beauty_std = df['beauty'].std()
allstudents_std = df['allstudents'].std()
results['model2_std_effect'] = results['model2_coef'] * (beauty_std / allstudents_std)
print('Model2 standardized effect', results['model2_std_effect'])

# Save results for potential inspection
pd.Series(results).to_csv('analysis_results.csv')
