import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('teachingratings.csv')

# Basic model: eval ~ beauty
m1 = smf.ols('eval ~ beauty', data=df).fit()

# Controlled model with common covariates
m2 = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students',
    data=df
).fit()

# Prepare summary stats
results = {
    'n': len(df),
    'beauty_mean': df['beauty'].mean(),
    'eval_mean': df['eval'].mean(),
    'm1_coef': m1.params['beauty'],
    'm1_p': m1.pvalues['beauty'],
    'm2_coef': m2.params['beauty'],
    'm2_p': m2.pvalues['beauty'],
}

# Save a short report
with open('analysis_report.txt', 'w') as f:
    f.write('Model 1 (eval ~ beauty):\n')
    f.write(m1.summary().as_text())
    f.write('\n\nModel 2 (with controls):\n')
    f.write(m2.summary().as_text())
    f.write('\n\nKey results:\n')
    for k, v in results.items():
        f.write(f'{k}: {v}\n')

# Also print key results for quick inspection
print(results)
