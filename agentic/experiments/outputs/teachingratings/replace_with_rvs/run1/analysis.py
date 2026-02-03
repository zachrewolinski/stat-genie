import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic model: eval on beauty
model_simple = smf.ols('eval ~ beauty', data=_df).fit()

# Control model with key covariates
# Using common controls from literature: age, gender, minority, native, tenure, division, credits, class size
# Convert categorical to category type for statsmodels
for col in ['gender','minority','native','tenure','division','credits']:
    _df[col] = _df[col].astype('category')

model_controls = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students',
    data=_df
).fit()

# Save key results to a small text block for reference
with open('analysis_results.txt','w') as f:
    f.write('Simple model:\n')
    f.write(str(model_simple.summary()) + '\n')
    f.write('\nControlled model:\n')
    f.write(str(model_controls.summary()) + '\n')

print('Simple model beauty coef:', model_simple.params['beauty'], 'p=', model_simple.pvalues['beauty'])
print('Controlled model beauty coef:', model_controls.params['beauty'], 'p=', model_controls.pvalues['beauty'])
