import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'caschools.csv'
df = pd.read_csv(path)

# Compute student-teacher ratio
# Avoid division by zero just in case
ratio = df['students'] / df['teachers']
df['stratio'] = ratio.replace([np.inf, -np.inf], np.nan)

# Academic performance: average of reading and math scores
# (both are standardized test scores in dataset)
df['avg_score'] = df[['read', 'math']].mean(axis=1)

# Drop rows with missing key values
analysis_df = df[['avg_score', 'stratio', 'income', 'lunch', 'english', 'calworks']].dropna()

# Correlation
corr = analysis_df['avg_score'].corr(analysis_df['stratio'])

# Simple OLS: avg_score ~ stratio
X_simple = sm.add_constant(analysis_df['stratio'])
model_simple = sm.OLS(analysis_df['avg_score'], X_simple).fit()

# Adjusted OLS with common socioeconomic controls
X_adj = analysis_df[['stratio', 'income', 'lunch', 'english', 'calworks']]
X_adj = sm.add_constant(X_adj)
model_adj = sm.OLS(analysis_df['avg_score'], X_adj).fit()

print('Rows used:', len(analysis_df))
print('Correlation (avg_score vs stratio):', corr)
print('\nSimple OLS: avg_score ~ stratio')
print(model_simple.summary().as_text())
print('\nAdjusted OLS: avg_score ~ stratio + income + lunch + english + calworks')
print(model_adj.summary().as_text())
