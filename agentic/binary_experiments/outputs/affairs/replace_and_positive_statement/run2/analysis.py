import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data
_df = pd.read_csv('affairs.csv')

# Basic group stats
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

summary = (
    _df.groupby('children')
    .agg(
        n=('affairs', 'size'),
        mean_affairs=('affairs', 'mean'),
        median_affairs=('affairs', 'median'),
        any_affair_rate=('any_affair', 'mean')
    )
)

# Two-sample t-test for mean affairs difference (no vs yes)
no_affairs = _df.loc[_df['children'] == 'no', 'affairs']
yes_affairs = _df.loc[_df['children'] == 'yes', 'affairs']

# Welch t-test
_t_stat, _p_val, _dfree = ttest_ind(no_affairs, yes_affairs, usevar='unequal')

# Logistic regression: any affair ~ children + controls
# Treat gender and children as categorical
logit_model = smf.logit(
    "any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=_df
).fit(disp=False)

# Poisson regression for count of affairs
poisson_model = smf.glm(
    "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=_df,
    family=sm.families.Poisson()
).fit()

# Extract key coefficients
children_coef_logit = logit_model.params.get('C(children)[T.yes]', np.nan)
children_p_logit = logit_model.pvalues.get('C(children)[T.yes]', np.nan)

children_coef_pois = poisson_model.params.get('C(children)[T.yes]', np.nan)
children_p_pois = poisson_model.pvalues.get('C(children)[T.yes]', np.nan)

# Summarize results
print('Group summary (children=no vs yes):')
print(summary)
print('\nWelch t-test for mean affairs (no - yes):')
print(f"t={_t_stat:.3f}, p={_p_val:.4f}, df={_dfree:.1f}")
print('\nLogit any_affair ~ children + controls:')
print(f"children_yes coef={children_coef_logit:.3f}, p={children_p_logit:.4f}")
print('\nPoisson affairs ~ children + controls:')
print(f"children_yes coef={children_coef_pois:.3f}, p={children_p_pois:.4f}")
