import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'caschools.csv'
df = pd.read_csv(path)

# Identify columns based on value ranges (shuffled names)
# english: 81-27176 -> student enrollment
# students: 4.85-1429 -> number of teachers
student_count = df['english']
teacher_count = df['students']

# Student-teacher ratio
stratio = student_count / teacher_count

# Academic performance: average of the two test score columns (~605-709)
score_cols = ['district', 'expenditure']
performance = df[score_cols].mean(axis=1)

# Simple correlation
corr = np.corrcoef(stratio, performance)[0, 1]

# OLS regression: performance on stratio
X = sm.add_constant(stratio)
model = sm.OLS(performance, X, missing='drop').fit()

results = {
    'n': int(model.nobs),
    'corr': float(corr),
    'coef_stratio': float(model.params.iloc[1]),
    'pvalue_stratio': float(model.pvalues.iloc[1]),
    'r2': float(model.rsquared),
}

print(results)
print(model.summary())
