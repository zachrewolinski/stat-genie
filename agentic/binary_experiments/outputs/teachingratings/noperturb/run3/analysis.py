import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic stats
summary = df[['beauty', 'eval']].describe()

# Simple correlation
corr = df['beauty'].corr(df['eval'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=df).fit()

# OLS with controls commonly used in literature
# Convert categorical variables via formula interface
model_controls = smf.ols(
    'eval ~ beauty + age + gender + minority + credits + division + native + tenure + students',
    data=df
).fit()

# Cluster-robust SE at professor level (multiple courses per professor)
# Use prof as clustering variable
model_controls_cluster = model_controls.get_robustcov_results(cov_type='cluster', groups=df['prof'])

# Print results
print('Summary stats (beauty, eval):')
print(summary)
print('\nCorrelation (beauty, eval):', corr)

print('\nSimple OLS: eval ~ beauty')
print(model_simple.summary().tables[1])

print('\nOLS with controls (standard SE):')
print(model_controls.summary().tables[1])

print('\nOLS with controls (clustered SE by prof):')
print(model_controls_cluster.summary().tables[1])
