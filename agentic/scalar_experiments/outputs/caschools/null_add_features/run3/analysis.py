import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('caschools.csv')

# Construct student-teacher ratio
df['stratio'] = df['students'] / df['teachers']

# Academic performance: average of read and math scores
df['avgscore'] = df[['read', 'math']].mean(axis=1)

# Drop rows with missing values in key variables
df_model = df[['avgscore', 'stratio', 'income', 'english', 'lunch', 'calworks']].dropna()

# Fit linear regression of avgscore on stratio and controls
X = df_model[['stratio', 'income', 'english', 'lunch', 'calworks']]
X = sm.add_constant(X)
y = df_model['avgscore']
model = sm.OLS(y, X).fit()

# Extract coefficient on stratio (expect negative if lower ratio -> higher performance)
coef = model.params['stratio']

# Map coefficient to Likert scale: more negative -> stronger evidence yes
se = model.bse['stratio']
t_value = coef / se if se != 0 else 0.0

# Convert t-stat into [-100, 100] with saturation beyond |t| >= 5
scaled = max(-5.0, min(5.0, -t_value))  # minus so that negative coef -> positive evidence
likert = int(round((scaled / 5.0) * 100))

with open('conclusion.txt', 'w') as f:
    f.write(str(likert))

print('Coefficient on stratio:', coef)
print('t-statistic:', t_value)
print('Likert score:', likert)
