import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
file_path = 'affairs.csv'
df = pd.read_csv(file_path)

# Prepare variables
# feature2: engagement in extramarital affairs (numeric, anonymized)
# feature6: children in marriage (yes/no)

df = df.dropna(subset=['feature2', 'feature6']).copy()
df['has_children'] = df['feature6'].map({'yes': 1, 'no': 0})

# Basic group stats
summary = df.groupby('has_children')['feature2'].agg(['count', 'mean', 'median', 'std'])

# Two-sample Welch t-test
group_no = df.loc[df['has_children'] == 0, 'feature2']
group_yes = df.loc[df['has_children'] == 1, 'feature2']

ttest_res = stats.ttest_ind(group_no, group_yes, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
# Use alternative='two-sided' for general difference; we will inspect direction via group means
mwu_res = stats.mannwhitneyu(group_no, group_yes, alternative='two-sided')

# Effect size (Cohen's d)
# d = (mean_no - mean_yes) / pooled_sd
mean_no = group_no.mean()
mean_yes = group_yes.mean()
var_no = group_no.var(ddof=1)
var_yes = group_yes.var(ddof=1)
pooled_sd = np.sqrt(((group_no.size - 1)*var_no + (group_yes.size - 1)*var_yes) / (group_no.size + group_yes.size - 2))
cohens_d = (mean_no - mean_yes) / pooled_sd if pooled_sd > 0 else np.nan

# OLS regression with controls
# Use formula with categorical gender (feature3) and children (feature6)
# Other features are numeric
formula = 'feature2 ~ C(feature6) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(feature3)'
ols_model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

# Logistic regression for any affairs (>0)
df['any_affair'] = (df['feature2'] > 0).astype(int)
logit_formula = 'any_affair ~ C(feature6) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(feature3)'
logit_model = smf.logit(formula=logit_formula, data=df).fit(disp=False)

# Extract key results
children_coef_ols = ols_model.params.get('C(feature6)[T.yes]', np.nan)
children_p_ols = ols_model.pvalues.get('C(feature6)[T.yes]', np.nan)

children_coef_logit = logit_model.params.get('C(feature6)[T.yes]', np.nan)
children_p_logit = logit_model.pvalues.get('C(feature6)[T.yes]', np.nan)

# Convert logit coef to odds ratio
odds_ratio = np.exp(children_coef_logit) if pd.notnull(children_coef_logit) else np.nan

results = {
    'summary': summary.to_dict(),
    'ttest': {
        'statistic': ttest_res.statistic,
        'pvalue': ttest_res.pvalue,
    },
    'mannwhitney': {
        'statistic': mwu_res.statistic,
        'pvalue': mwu_res.pvalue,
    },
    'effect_size': {
        'cohens_d_no_minus_yes': cohens_d,
        'mean_no': mean_no,
        'mean_yes': mean_yes,
    },
    'ols': {
        'coef_children_yes': children_coef_ols,
        'pvalue_children_yes': children_p_ols,
        'r2': ols_model.rsquared,
    },
    'logit': {
        'coef_children_yes': children_coef_logit,
        'pvalue_children_yes': children_p_logit,
        'odds_ratio_children_yes': odds_ratio,
    },
    'counts': {
        'n_total': int(df.shape[0]),
        'n_children_yes': int((df['has_children'] == 1).sum()),
        'n_children_no': int((df['has_children'] == 0).sum()),
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
