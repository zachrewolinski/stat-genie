import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
DF_PATH = 'affairs.csv'
df = pd.read_csv(DF_PATH)

# Basic cleaning
# children is yes/no
if df['children'].dtype != object:
    df['children'] = df['children'].astype(str)

# Create binary indicators
child_yes = df['children'].str.lower().eq('yes').astype(int)

df['children_yes'] = child_yes

df['any_affair'] = (df['affairs'] > 0).astype(int)

def safe_mean(x):
    return float(np.mean(x))

# Group statistics
mean_affairs_yes = safe_mean(df.loc[df['children_yes'] == 1, 'affairs'])
mean_affairs_no = safe_mean(df.loc[df['children_yes'] == 0, 'affairs'])
prop_any_yes = safe_mean(df.loc[df['children_yes'] == 1, 'any_affair'])
prop_any_no = safe_mean(df.loc[df['children_yes'] == 0, 'any_affair'])

# T-test on affairs counts
# Unequal variances
res_t = stats.ttest_ind(
    df.loc[df['children_yes'] == 1, 'affairs'],
    df.loc[df['children_yes'] == 0, 'affairs'],
    equal_var=False,
    nan_policy='omit'
)

# Logistic regression for any affairs with controls
# Use common Fair dataset controls
controls = ['gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
# Prepare design matrix
X = df[['children_yes'] + controls].copy()
# Encode categorical gender
X = pd.get_dummies(X, columns=['gender'], drop_first=True)
X = sm.add_constant(X, has_constant='add')

y = df['any_affair']
logit_model = sm.Logit(y, X)
try:
    logit_res = logit_model.fit(disp=False)
    logit_coef = logit_res.params['children_yes']
    logit_p = logit_res.pvalues['children_yes']
except Exception as e:
    logit_coef = np.nan
    logit_p = np.nan

# OLS on log1p(affairs) with controls
ols_y = np.log1p(df['affairs'])
ols_model = sm.OLS(ols_y, X)
try:
    ols_res = ols_model.fit()
    ols_coef = ols_res.params['children_yes']
    ols_p = ols_res.pvalues['children_yes']
except Exception as e:
    ols_coef = np.nan
    ols_p = np.nan

# Summarize
summary = {
    'n': int(df.shape[0]),
    'mean_affairs_yes': mean_affairs_yes,
    'mean_affairs_no': mean_affairs_no,
    'mean_diff_yes_minus_no': mean_affairs_yes - mean_affairs_no,
    'prop_any_yes': prop_any_yes,
    'prop_any_no': prop_any_no,
    'prop_diff_yes_minus_no': prop_any_yes - prop_any_no,
    't_stat': float(res_t.statistic),
    't_pvalue': float(res_t.pvalue),
    'logit_children_coef': float(logit_coef),
    'logit_children_p': float(logit_p),
    'ols_log1p_children_coef': float(ols_coef),
    'ols_log1p_children_p': float(ols_p),
}

print(summary)
