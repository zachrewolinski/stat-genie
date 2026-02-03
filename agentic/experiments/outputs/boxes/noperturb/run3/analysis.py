import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import Table
import statsmodels.api as sm
from scipy.stats import chi2

# Load data
df = pd.read_csv('boxes.csv')

# Basic recodes
df['majority_choice'] = (df['y'] == 2).astype(int)
df['minority_choice'] = (df['y'] == 3).astype(int)
df['unchosen_choice'] = (df['y'] == 1).astype(int)

# Age groups for developmental stages
age_bins = [3, 6, 9, 12, 14]
age_labels = ['4-6', '7-9', '10-12', '13-14']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True, include_lowest=True)

# Crosstabs and chi-square tests
def chi2_test(ct):
    table = Table(ct.values)
    res = table.test_nominal_association()
    return res.statistic, res.pvalue, res.df

culture_ct = pd.crosstab(df['culture'], df['y'])
age_ct = pd.crosstab(df['age_group'], df['y'])

culture_chi2, culture_p, culture_df = chi2_test(culture_ct)
age_chi2, age_p, age_df = chi2_test(age_ct)

# Multinomial logit: y on age + culture + interaction
# Build design matrix
y = df['y']
culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)
age = df['age']
# Interaction terms: age * culture dummies
interaction = culture_dummies.mul(age, axis=0)
interaction.columns = [c + '_x_age' for c in interaction.columns]
X = pd.concat([age.rename('age'), culture_dummies, interaction], axis=1)
X = sm.add_constant(X)

# Fit model
mnlogit = sm.MNLogit(y, X)
try:
    mn_res = mnlogit.fit(method='newton', disp=False, maxiter=200)
    mn_summary = mn_res.summary()
    llf = mn_res.llf
    df_model = mn_res.df_model
except Exception as e:
    mn_res = None
    mn_summary = str(e)
    llf = np.nan
    df_model = np.nan

# Model without interaction for comparison
X_no_inter = pd.concat([age.rename('age'), culture_dummies], axis=1)
X_no_inter = sm.add_constant(X_no_inter)
mnlogit_no_inter = sm.MNLogit(y, X_no_inter)
try:
    mn_res_no_inter = mnlogit_no_inter.fit(method='newton', disp=False, maxiter=200)
    llf_no_inter = mn_res_no_inter.llf
    df_model_no_inter = mn_res_no_inter.df_model
except Exception as e:
    mn_res_no_inter = None
    llf_no_inter = np.nan
    df_model_no_inter = np.nan

# Likelihood ratio test for interaction
if mn_res is not None and mn_res_no_inter is not None:
    lr_stat = 2 * (llf - llf_no_inter)
    lr_df = df_model - df_model_no_inter
else:
    lr_stat = np.nan
    lr_df = np.nan

# Compute proportions by culture and age group
def proportion_table(group_col):
    ct = pd.crosstab(df[group_col], df['y'], normalize='index')
    return ct

culture_props = proportion_table('culture')
age_props = proportion_table('age_group')

# Output results
print('Chi-square culture vs choice:')
print(f'chi2={culture_chi2:.3f}, df={culture_df}, p={culture_p:.6f}')
print('\nChi-square age group vs choice:')
print(f'chi2={age_chi2:.3f}, df={age_df}, p={age_p:.6f}')

print('\nMajority-choice proportion by culture:')
print(culture_props[2].round(3).to_string())

print('\nMajority-choice proportion by age group:')
print(age_props[2].round(3).to_string())

if mn_res is not None:
    print('\nMultinomial logit with age, culture, and interaction fitted.')
    print(f'LL(full)={llf:.3f}, df_model(full)={df_model}')
    print(f'LL(no interaction)={llf_no_inter:.3f}, df_model(no_inter)={df_model_no_inter}')
    lr_p = chi2.sf(lr_stat, lr_df)
    print(f'LR stat for interaction={lr_stat:.3f}, df={lr_df}, p={lr_p:.6f}')
else:
    print('\nMultinomial logit failed:', mn_summary)
