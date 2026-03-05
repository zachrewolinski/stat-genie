import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Columns
# feature2: affair frequency
# feature6: children yes/no

# Clean
# Ensure feature6 lower
if df['feature6'].dtype == object:
    df['feature6'] = df['feature6'].str.lower()

# Create binary: any affair

df['any_affair'] = (df['feature2'] > 0).astype(int)

# Group summaries
summary = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])
summary_any = df.groupby('feature6')['any_affair'].agg(['mean','sum','count'])

# Mann-Whitney U test (two-sided)
# Need samples
children_yes = df.loc[df['feature6']=='yes','feature2']
children_no = df.loc[df['feature6']=='no','feature2']

mw = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')

# t-test (Welch)
welch = stats.ttest_ind(children_yes, children_no, equal_var=False)

# Chi-square for any_affair
cont = pd.crosstab(df['feature6'], df['any_affair'])
chi2 = stats.chi2_contingency(cont)

# Logistic regression with controls
# Use C() for categorical
formula = 'any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
model = smf.logit(formula, data=df).fit(disp=False)

# Extract coefficient for children yes relative to no
# In statsmodels, C(feature6)[T.yes] if 'no' is reference. We'll inspect.
params = model.params
pvalues = model.pvalues

coef_children = params.get('C(feature6)[T.yes]', np.nan)
pval_children = pvalues.get('C(feature6)[T.yes]', np.nan)

# Odds ratio
or_children = float(np.exp(coef_children)) if pd.notnull(coef_children) else np.nan

# Output
print('Summary affair frequency by children:')
print(summary)
print('\nProportion any affair by children:')
print(summary_any)
print('\nMann-Whitney U:', mw)
print('Welch t-test:', welch)
print('\nChi-square any_affair:', chi2)
print('\nLogit coef children yes (vs no):', coef_children, 'OR', or_children, 'p', pval_children)
print('\nLogit summary:')
print(model.summary())
