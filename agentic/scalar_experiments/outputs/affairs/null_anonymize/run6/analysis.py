import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map children: yes=1 no=0
children = _df['feature6'].map({'yes': 1, 'no': 0})

affairs = _df['feature2'].astype(float)

# Group stats
mean_yes = affairs[children == 1].mean()
mean_no = affairs[children == 0].mean()

# Any affair
any_affair = (affairs > 0).astype(int)
prop_yes = any_affair[children == 1].mean()
prop_no = any_affair[children == 0].mean()

# Welch t-test
welch = stats.ttest_ind(affairs[children == 1], affairs[children == 0], equal_var=False)

# Mann-Whitney U (two-sided)
try:
    mwu = stats.mannwhitneyu(affairs[children == 1], affairs[children == 0], alternative='two-sided')
except Exception:
    mwu = None

# Logistic regression: any_affair ~ children
X = sm.add_constant(children)
logit = sm.Logit(any_affair, X).fit(disp=False)

# Extract odds ratio and p-value for children
coef = logit.params['feature6'] if 'feature6' in logit.params else logit.params[1]
# handle column names
if 'feature6' in logit.params.index:
    coef = logit.params['feature6']
    pval = logit.pvalues['feature6']
else:
    coef = logit.params.iloc[1]
    pval = logit.pvalues.iloc[1]

odds_ratio = float(np.exp(coef))

# Save results
results = {
    'mean_affairs_children_yes': float(mean_yes),
    'mean_affairs_children_no': float(mean_no),
    'prop_any_affair_children_yes': float(prop_yes),
    'prop_any_affair_children_no': float(prop_no),
    'welch_t_stat': float(welch.statistic),
    'welch_p_value': float(welch.pvalue),
    'mannwhitney_u_stat': float(mwu.statistic) if mwu is not None else None,
    'mannwhitney_p_value': float(mwu.pvalue) if mwu is not None else None,
    'logit_odds_ratio_children_yes': odds_ratio,
    'logit_p_value': float(pval),
    'n_yes': int((children == 1).sum()),
    'n_no': int((children == 0).sum())
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
