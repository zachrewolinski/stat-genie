import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data

df = pd.read_csv('affairs.csv')

# Binary indicators

df['children_yes'] = (df['children'] == 'yes').astype(int)
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Descriptive stats

group_means = df.groupby('children')['affairs'].mean()
group_any = df.groupby('children')['any_affair'].mean()

# Two-sample t-test on affairs count by children

aff_yes = df.loc[df['children'] == 'yes', 'affairs']
aff_no = df.loc[df['children'] == 'no', 'affairs']

t_stat, p_val, _ = ttest_ind(aff_yes, aff_no, usevar='unequal')

# Logistic regression for any affair with controls

logit_model = smf.logit(
    'any_affair ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=df
).fit(disp=False)

# Poisson regression for affair counts with controls

poisson_model = smf.glm(
    'affairs ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=df,
    family=sm.families.Poisson()
).fit(cov_type='HC1')

# Collect results
results = {
    'mean_affairs_children_yes': group_means.get('yes'),
    'mean_affairs_children_no': group_means.get('no'),
    'share_any_affair_children_yes': group_any.get('yes'),
    'share_any_affair_children_no': group_any.get('no'),
    't_stat': float(t_stat),
    't_p_value': float(p_val),
    'logit_children_coef': float(logit_model.params['children_yes']),
    'logit_children_p': float(logit_model.pvalues['children_yes']),
    'poisson_children_coef': float(poisson_model.params['children_yes']),
    'poisson_children_p': float(poisson_model.pvalues['children_yes']),
}

print('RESULTS')
for k, v in results.items():
    print(f'{k}: {v}')
