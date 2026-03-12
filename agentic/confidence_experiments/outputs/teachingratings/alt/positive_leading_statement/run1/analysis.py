import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic stats
n = len(df)

# Simple correlation
corr = df[['beauty','eval']].corr().iloc[0,1]

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Add controls
# Use categorical variables as C()
# Use log of students and allstudents for scale.
# Include age and age^2 to allow nonlinearity.

df = df.copy()
df['log_students'] = np.log(df['students'])
df['log_allstudents'] = np.log(df['allstudents'])
df['age2'] = df['age']**2

formula_controls = (
    'eval ~ beauty + age + age2 + C(gender) + C(minority) + C(native) + C(tenure) '
    '+ C(division) + C(credits) + log_students + log_allstudents'
)
model_controls = smf.ols(formula_controls, data=df).fit(cov_type='HC3')

# Effect size: predicted change in eval for 1 SD increase in beauty
beauty_sd = df['beauty'].std()
coef = model_controls.params['beauty']

# Build results summary for later reasoning
results = {
    'n': int(n),
    'corr_beauty_eval': float(corr),
    'simple': {
        'coef': float(model_simple.params['beauty']),
        'se': float(model_simple.bse['beauty']),
        'pvalue': float(model_simple.pvalues['beauty']),
        'r2': float(model_simple.rsquared)
    },
    'controls': {
        'coef': float(coef),
        'se': float(model_controls.bse['beauty']),
        'pvalue': float(model_controls.pvalues['beauty']),
        'r2': float(model_controls.rsquared)
    },
    'beauty_sd': float(beauty_sd),
    'eval_sd': float(df['eval'].std()),
    'effect_1sd_beauty': float(coef * beauty_sd)
}

print(json.dumps(results, indent=2))
