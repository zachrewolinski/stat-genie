import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('teachingratings.csv')

# Basic checks
# Build design matrices
# Categorical columns
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']

# Use students (participants) as class size control; age numeric
num_cols = ['beauty', 'age', 'students']

# Prepare data
X = df[num_cols].copy()
X = pd.concat([X, pd.get_dummies(df[cat_cols], drop_first=True)], axis=1)
X = sm.add_constant(X)

y = df['eval']

# Model 1: eval ~ beauty
X1 = sm.add_constant(df[['beauty']])
model1 = sm.OLS(y, X1).fit(cov_type='HC1')

# Model 2: with controls
model2 = sm.OLS(y, X).fit(cov_type='HC1')

# Summaries
summary = {
    'n': len(df),
    'model1_beauty_coef': model1.params['beauty'],
    'model1_beauty_p': model1.pvalues['beauty'],
    'model2_beauty_coef': model2.params['beauty'],
    'model2_beauty_p': model2.pvalues['beauty'],
    'model2_r2': model2.rsquared,
}

print(summary)
