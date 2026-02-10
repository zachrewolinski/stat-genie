import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('caschools.csv')

# Create student-teacher ratio
df['stratio'] = df['students'] / df['teachers']

# Outcome: average of read and math
df['avgscore'] = df[['read', 'math']].mean(axis=1)

# Simple bivariate regression
X = sm.add_constant(df['stratio'])
y = df['avgscore']
model = sm.OLS(y, X).fit()

print('N =', len(df))
print(model.summary())

# Also compute correlation
corr = df['stratio'].corr(df['avgscore'])
print('Correlation:', corr)

# Save key results for later use
with open('analysis_results.txt', 'w') as f:
    f.write(f"coef_stratio={model.params['stratio']}\n")
    f.write(f"pvalue_stratio={model.pvalues['stratio']}\n")
    f.write(f"corr={corr}\n")
