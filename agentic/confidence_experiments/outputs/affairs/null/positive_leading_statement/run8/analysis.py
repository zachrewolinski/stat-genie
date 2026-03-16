import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

# Load data
df = pd.read_csv('affairs.csv')

# Binary indicator for any affair
df['had_affair'] = (df['affairs'] > 0).astype(int)

# Basic group stats
summary = {}

n_total = len(df)
summary['n_total'] = n_total

children_counts = df['children'].value_counts().to_dict()
summary['children_counts'] = children_counts

rate_by_children = df.groupby('children')['had_affair'].mean().to_dict()
summary['affair_rate_by_children'] = rate_by_children

mean_affairs_by_children = df.groupby('children')['affairs'].mean().to_dict()
summary['mean_affairs_by_children'] = mean_affairs_by_children

# Difference in proportions test (any affair vs none)
counts = df.groupby('children')['had_affair'].sum()
ns = df.groupby('children')['had_affair'].count()

# Ensure order no, yes for interpretability
count_no, count_yes = counts['no'], counts['yes']
n_no, n_yes = ns['no'], ns['yes']

stat, pval = proportions_ztest([count_yes, count_no], [n_yes, n_no])
summary['prop_ztest'] = {
    'z_stat': float(stat),
    'p_value': float(pval),
}

# Logistic regression: any affair on children only
logit_simple = smf.logit('had_affair ~ C(children)', data=df).fit(disp=False)
params = logit_simple.params
conf_int = logit_simple.conf_int()
pvalues = logit_simple.pvalues

coef_children_yes = float(params.get('C(children)[T.yes]', float('nan')))
p_children_yes = float(pvalues.get('C(children)[T.yes]', float('nan')))
ci_low, ci_high = conf_int.loc['C(children)[T.yes]']

summary['logit_children_only'] = {
    'coef_children_yes': coef_children_yes,
    'p_value_children_yes': p_children_yes,
    'ci_low_children_yes': float(ci_low),
    'ci_high_children_yes': float(ci_high),
}

# Logistic regression with controls (age, yearsmarried, rating, religiousness, gender)
logit_full = smf.logit(
    'had_affair ~ C(children) + age + yearsmarried + rating + religiousness + C(gender)',
    data=df,
).fit(disp=False)

params_f = logit_full.params
conf_int_f = logit_full.conf_int()
pvalues_f = logit_full.pvalues

coef_children_yes_f = float(params_f.get('C(children)[T.yes]', float('nan')))
p_children_yes_f = float(pvalues_f.get('C(children)[T.yes]', float('nan')))
ci_low_f, ci_high_f = conf_int_f.loc['C(children)[T.yes]']

summary['logit_with_controls'] = {
    'coef_children_yes': coef_children_yes_f,
    'p_value_children_yes': p_children_yes_f,
    'ci_low_children_yes': float(ci_low_f),
    'ci_high_children_yes': float(ci_high_f),
}

# Print summary in a readable form
import json
print(json.dumps(summary, indent=2))
