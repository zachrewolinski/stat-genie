import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Encode children yes/no
children = df['feature6'].str.strip().str.lower().map({'yes': 1, 'no': 0})
df = df.copy()
df['children'] = children

# Affair intensity
aff = df['feature2']

# Basic group stats
stats = df.groupby('children')['feature2'].agg(['mean', 'median', 'count'])

# Proportion with any affairs
any_aff = (aff > 0).astype(int)
df['any_aff'] = any_aff
rate = df.groupby('children')['any_aff'].mean()

# Difference in means
mean_no = stats.loc[0, 'mean']
mean_yes = stats.loc[1, 'mean']
mean_diff = mean_yes - mean_no

# Difference in rates
rate_no = rate.loc[0]
rate_yes = rate.loc[1]
rate_diff = rate_yes - rate_no

# Effect size (Cohen's d) for mean difference
std_pooled = np.sqrt((
    df.loc[df['children'] == 0, 'feature2'].var(ddof=1) +
    df.loc[df['children'] == 1, 'feature2'].var(ddof=1)
) / 2)
cohen_d = (mean_yes - mean_no) / std_pooled if std_pooled > 0 else np.nan

# Regression with controls: feature2 on children + controls
reg_df = df[['feature2', 'children', 'feature3', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']].copy()
reg_df['female'] = reg_df['feature3'].str.strip().str.lower().map({'female': 1, 'male': 0})

X = reg_df[['children', 'female', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']]
X = sm.add_constant(X)
model = sm.OLS(reg_df['feature2'], X, missing='drop').fit()

# Linear probability model on any affairs
reg_df2 = reg_df.copy()
reg_df2['any_aff'] = (reg_df2['feature2'] > 0).astype(int)
X2 = sm.add_constant(reg_df2[['children', 'female', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']])
model2 = sm.OLS(reg_df2['any_aff'], X2, missing='drop').fit()

# Save summary stats to file for inspection
with open('analysis_summary.txt', 'w') as f:
    f.write('Group stats (feature2) by children (0=no,1=yes)\n')
    f.write(str(stats) + '\n\n')
    f.write('Rate any affairs by children (0=no,1=yes)\n')
    f.write(str(rate) + '\n\n')
    f.write(f'mean_diff (yes-no): {mean_diff:.4f}\n')
    f.write(f'rate_diff (yes-no): {rate_diff:.4f}\n')
    f.write(f'cohen_d: {cohen_d:.4f}\n\n')
    f.write('OLS on feature2:\n')
    f.write(model.summary().as_text() + '\n\n')
    f.write('OLS on any_aff:\n')
    f.write(model2.summary().as_text() + '\n')

print('done')
