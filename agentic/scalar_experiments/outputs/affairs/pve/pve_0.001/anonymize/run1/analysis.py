import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Basic cleaning
# Ensure feature6 is categorical yes/no
# feature2 is numeric engagement measure

# Drop rows with missing key fields
key_cols = ['feature2', 'feature6']
df_key = df.dropna(subset=key_cols).copy()

# Map children yes/no
children = df_key['feature6'].astype(str).str.lower().str.strip()
# Keep only yes/no
mask = children.isin(['yes', 'no'])
df_key = df_key.loc[mask].copy()
children = children.loc[mask]

# Create binary: 1 if children yes
children_yes = (children == 'yes').astype(int)
df_key['children_yes'] = children_yes

# Engagement variable
eng = pd.to_numeric(df_key['feature2'], errors='coerce')
eng = eng.loc[eng.notna()]
# align df
aligned = df_key.loc[eng.index].copy()
aligned['eng'] = eng

# Descriptive stats
summary = aligned.groupby('children_yes')['eng'].agg(['count','mean','std','median'])

# Difference in means (yes - no)
mean_yes = summary.loc[1, 'mean'] if 1 in summary.index else np.nan
mean_no = summary.loc[0, 'mean'] if 0 in summary.index else np.nan
mean_diff = mean_yes - mean_no

# t-test (unequal variances)
eng_yes = aligned.loc[aligned['children_yes'] == 1, 'eng']
eng_no = aligned.loc[aligned['children_yes'] == 0, 'eng']

t_stat, p_val = stats.ttest_ind(eng_yes, eng_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparametric)
try:
    u_stat, p_u = stats.mannwhitneyu(eng_yes, eng_no, alternative='two-sided')
except Exception:
    p_u = np.nan

# Effect size (Cohen's d)
# pooled std (unequal n)
var_yes = np.nanvar(eng_yes, ddof=1)
var_no = np.nanvar(eng_no, ddof=1)

n_yes = eng_yes.shape[0]
n_no = eng_no.shape[0]
pooled_sd = np.sqrt(((n_yes-1)*var_yes + (n_no-1)*var_no) / (n_yes + n_no - 2)) if (n_yes + n_no - 2) > 0 else np.nan
cohens_d = mean_diff / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

# Regression with controls (linear)
# Use available features as controls (age, years married, relig, education, occupation, marriage rating, gender)
# Convert gender to binary
reg_df = aligned.copy()
reg_df['gender_male'] = reg_df['feature3'].astype(str).str.lower().str.strip().map({'male':1, 'female':0})

# Use numeric columns for controls
control_cols = ['feature4','feature5','feature7','feature8','feature9','feature10','gender_male']

# Coerce numeric
for col in control_cols:
    reg_df[col] = pd.to_numeric(reg_df[col], errors='coerce')

reg_df = reg_df.dropna(subset=['eng','children_yes'] + control_cols)

if len(reg_df) > 0:
    X = reg_df[['children_yes'] + control_cols]
    X = sm.add_constant(X, has_constant='add')
    y = reg_df['eng']
    model = sm.OLS(y, X).fit(cov_type='HC3')
    coef = model.params.get('children_yes', np.nan)
    p_reg = model.pvalues.get('children_yes', np.nan)
else:
    coef = np.nan
    p_reg = np.nan

results = {
    'n_total': int(len(aligned)),
    'n_yes': int(n_yes),
    'n_no': int(n_no),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'mean_diff_yes_minus_no': float(mean_diff),
    't_p_value': float(p_val),
    'mw_p_value': float(p_u) if p_u==p_u else None,
    'cohens_d': float(cohens_d) if cohens_d==cohens_d else None,
    'reg_coef_children': float(coef) if coef==coef else None,
    'reg_p_value': float(p_reg) if p_reg==p_reg else None,
}

with open('analysis_results.json','w') as f:
    json.dump(results,f,indent=2)

print(json.dumps(results, indent=2))
