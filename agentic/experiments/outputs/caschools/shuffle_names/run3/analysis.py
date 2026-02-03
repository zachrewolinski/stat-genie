import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('caschools.csv')

# Map shuffled columns based on metadata and observed distributions
# english -> enrollment (students)
# students -> number of teachers
# district -> average reading score
# expenditure -> average math score

# Student-teacher ratio
str_ratio = df['english'] / df['students']

# Academic performance: average of reading and math scores
score = (df['district'] + df['expenditure']) / 2

# Correlation
corr = score.corr(str_ratio)

# Regression: score on STR
X = sm.add_constant(str_ratio)
model = sm.OLS(score, X).fit()

print('Student-teacher ratio summary')
print(str_ratio.describe())
print('\nTest score summary')
print(score.describe())
print(f'\nCorrelation (score vs STR): {corr:.4f}')
print('\nOLS regression: score ~ STR')
print(model.summary())
