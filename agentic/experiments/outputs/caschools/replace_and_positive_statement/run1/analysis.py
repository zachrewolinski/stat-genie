import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'caschools.csv'
df = pd.read_csv(csv_path)

# Compute student-teacher ratio and average score
# teachers appears as FTE counts; ratio = students / teachers
# Guard against division by zero though none expected

df = df.copy()
df['str'] = df['students'] / df['teachers']
df['avgscore'] = df[['read', 'math']].mean(axis=1)

# Simple correlation
corr = df[['str', 'avgscore']].corr().iloc[0, 1]

# Simple regression avgscore ~ str
X1 = sm.add_constant(df['str'])
model1 = sm.OLS(df['avgscore'], X1).fit()

# Multiple regression with common controls
controls = ['income', 'lunch', 'english', 'expenditure', 'computer', 'calworks']
X2 = sm.add_constant(df[['str'] + controls])
model2 = sm.OLS(df['avgscore'], X2).fit()

# Prepare concise results
results = {
    'n': int(df.shape[0]),
    'corr_str_avgscore': float(corr),
    'model1_coef_str': float(model1.params['str']),
    'model1_pvalue_str': float(model1.pvalues['str']),
    'model1_r2': float(model1.rsquared),
    'model2_coef_str': float(model2.params['str']),
    'model2_pvalue_str': float(model2.pvalues['str']),
    'model2_r2': float(model2.rsquared),
}

print('Summary results:')
for k, v in results.items():
    print(f'{k}: {v}')
