import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Clean: ensure children and gender categories
_df['children'] = _df['children'].astype('category')
_df['gender'] = _df['gender'].astype('category')

# Basic group stats
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

summary = _df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    prop_any=('any_affair','mean')
)

# t-test on affairs count
children_yes = _df.loc[_df['children']=='yes','affairs']
children_no = _df.loc[_df['children']=='no','affairs']

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False)

# Logistic regression for any affair
logit_model = smf.logit('any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=_df).fit(disp=False)

# Negative binomial for affair counts (adds 1e-8 to avoid issues? not needed)
nb_model = smf.glm('affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
                  data=_df, family=sm.families.NegativeBinomial()).fit()

# Extract effect for children yes vs no
# In statsmodels, C(children)[T.yes] if reference is no.

logit_coef = logit_model.params.get('C(children)[T.yes]')
logit_se = logit_model.bse.get('C(children)[T.yes]')
logit_p = logit_model.pvalues.get('C(children)[T.yes]')

nb_coef = nb_model.params.get('C(children)[T.yes]')
nb_se = nb_model.bse.get('C(children)[T.yes]')
nb_p = nb_model.pvalues.get('C(children)[T.yes]')

# Convert to odds ratio / incidence rate ratio
logit_or = float(np.exp(logit_coef)) if logit_coef is not None else np.nan
nb_irr = float(np.exp(nb_coef)) if nb_coef is not None else np.nan

results = {
    'summary': summary,
    'ttest_stat': ttest.statistic,
    'ttest_p': ttest.pvalue,
    'logit_coef': logit_coef,
    'logit_se': logit_se,
    'logit_p': logit_p,
    'logit_or': logit_or,
    'nb_coef': nb_coef,
    'nb_se': nb_se,
    'nb_p': nb_p,
    'nb_irr': nb_irr,
}

# Save key results for review
with open('analysis_results.txt','w') as f:
    f.write('Group summary:\n')
    f.write(summary.to_string())
    f.write('\n\nT-test affairs (yes vs no):\n')
    f.write(f'stat={ttest.statistic:.4f}, p={ttest.pvalue:.4g}\n')
    f.write('\nLogit any_affair ~ children + controls:\n')
    f.write(f'coef={logit_coef:.4f}, se={logit_se:.4f}, p={logit_p:.4g}, OR={logit_or:.4f}\n')
    f.write('\nNegBin affairs ~ children + controls:\n')
    f.write(f'coef={nb_coef:.4f}, se={nb_se:.4f}, p={nb_p:.4g}, IRR={nb_irr:.4f}\n')
