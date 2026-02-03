import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'caschools.csv'
df = pd.read_csv(csv_path)

# Define variables based on metadata
# Student-teacher ratio = total enrollment / number of teachers
student_teacher_ratio = df['feature6'] / df['feature7']

# Academic performance: average of reading and math scores
performance = (df['feature14'] + df['feature15']) / 2.0

# Simple correlation
corr = np.corrcoef(student_teacher_ratio, performance)[0, 1]

# Linear regression: performance ~ student_teacher_ratio
X = sm.add_constant(student_teacher_ratio)
model = sm.OLS(performance, X).fit()

# Save key results for interpretation
results = {
    'correlation': float(corr),
    'slope': float(model.params.iloc[1]),
    'slope_pvalue': float(model.pvalues.iloc[1]),
    'r_squared': float(model.rsquared),
    'n': int(len(df))
}

print('Results:', results)
