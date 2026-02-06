import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

# Load data
_df = pd.read_csv('affairs.csv')

# Prepare variables
_df['children_yes'] = (_df['children'].str.lower() == 'yes').astype(int)

# Descriptive stats
means = _df.groupby('children')['affairs'].mean()
mean_yes = means.get('yes', float('nan'))
mean_no = means.get('no', float('nan'))
mean_diff = mean_yes - mean_no  # negative means fewer affairs with children

# t-test (unequal variances)
stat, p_value, dfree = ttest_ind(
    _df.loc[_df['children'] == 'yes', 'affairs'],
    _df.loc[_df['children'] == 'no', 'affairs'],
    usevar='unequal'
)

# OLS with controls
ols_model = smf.ols(
    'affairs ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=_df
).fit(cov_type='HC3')

# Poisson GLM with controls (count outcome)
poisson_model = smf.glm(
    'affairs ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=_df,
    family=sm.families.Poisson()
).fit(cov_type='HC3')

# Extract coefficients and p-values for children
ols_coef = ols_model.params['children_yes']
ols_p = ols_model.pvalues['children_yes']
pois_coef = poisson_model.params['children_yes']
pois_p = poisson_model.pvalues['children_yes']

# Save key results for reporting
summary = {
    'mean_affairs_children_yes': float(mean_yes),
    'mean_affairs_children_no': float(mean_no),
    'mean_diff_yes_minus_no': float(mean_diff),
    'ttest_stat': float(stat),
    'ttest_p': float(p_value),
    'ols_children_coef': float(ols_coef),
    'ols_children_p': float(ols_p),
    'poisson_children_coef': float(pois_coef),
    'poisson_children_p': float(pois_p)
}

print(summary)
