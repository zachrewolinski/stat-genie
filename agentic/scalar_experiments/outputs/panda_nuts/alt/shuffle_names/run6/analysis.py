import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Reconstruct correct variable meanings based on metadata mismatch
# Column mapping inferred from values
# age -> age (years)
# nuts_opened (categorical m/f) -> sex
# seconds (categorical y/N) -> help (received help)
# help (numeric) -> nuts_opened
# chimpanzee (numeric) -> seconds

# Rename for clarity
rename_map = {
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'seconds',
    'seconds': 'help'
}

df = df.rename(columns=rename_map)

# Clean types
# help column is categorical y/N; normalize to 1/0
# some entries might be lowercase/uppercase

df['help'] = df['help'].astype(str).str.strip().str.lower().map({'y': 1, 'n': 0})

# Sex mapping m/f
# keep as categorical

# Compute efficiency = nuts opened per second
# avoid division by zero (if any)

# ensure numeric
for col in ['age', 'hammer', 'nuts_opened', 'seconds']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Drop rows with missing key values
analysis_df = df.dropna(subset=['efficiency', 'age', 'sex', 'help'])

# Basic descriptive
print('N rows:', len(analysis_df))
print('Efficiency summary:', analysis_df['efficiency'].describe())
print('N zeros nuts opened:', (analysis_df['nuts_opened'] == 0).sum())
print('Help counts:', analysis_df['help'].value_counts(dropna=False).to_dict())
print('Sex counts:', analysis_df['sex'].value_counts(dropna=False).to_dict())

# Regression model: efficiency ~ age + sex + help
# Use OLS with robust SE (HC3)
model = smf.ols('efficiency ~ age + C(sex) + help', data=analysis_df).fit(cov_type='HC3')
print(model.summary())

# Also test log1p efficiency to reduce skew (handle zeros)
analysis_df['log_eff'] = np.log1p(analysis_df['efficiency'])
model_log = smf.ols('log_eff ~ age + C(sex) + help', data=analysis_df).fit(cov_type='HC3')
print('\nLOG1P efficiency model')
print(model_log.summary())

# ANOVA / partial effects? We'll compute effect sizes for predictors using standardized beta
from sklearn.preprocessing import StandardScaler

X = analysis_df[['age', 'help']].copy()
# add sex dummies
sex_dummies = pd.get_dummies(analysis_df['sex'], drop_first=True)
X = pd.concat([X, sex_dummies], axis=1)

# standardize predictors and outcome
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
Y_std = StandardScaler().fit_transform(analysis_df[['efficiency']]).ravel()
X_std = sm.add_constant(X_std)
std_model = sm.OLS(Y_std, X_std).fit(cov_type='HC3')
print('\nStandardized model coefficients (z-scored):')
print(std_model.params)

# Output p-values for key predictors
print('\nP-values (efficiency model):')
print(model.pvalues)
print('\nP-values (log efficiency model):')
print(model_log.pvalues)

# Check correlation between age and efficiency
corr = analysis_df[['age','efficiency']].corr().iloc[0,1]
print('\nCorrelation age-efficiency:', corr)
