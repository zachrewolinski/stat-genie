import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path('affairs.csv')

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure children is categorical with 'no' as reference

df['children'] = df['children'].astype('category')

# Create binary affair indicator

df['affair_any'] = (df['affairs'] > 0).astype(int)

# Descriptive stats by children
summary = df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    affair_any_rate=('affair_any', 'mean')
)

# Difference in means t-test (Welch)
from scipy import stats

children_yes = df[df['children'] == 'yes']['affairs']
children_no = df[df['children'] == 'no']['affairs']

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False)

# OLS regression on affairs (as continuous) with controls
# Use robust (HC3) SE
ols_model = smf.ols(
    'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df
).fit(cov_type='HC3')

# Logistic regression on any affair with controls
logit_model = smf.logit(
    'affair_any ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df
).fit(disp=False)

# Marginal effect for children (yes) on probability in logit
margeff = logit_model.get_margeff(at='overall').summary_frame()

# Extract key effects
ols_children_coef = ols_model.params.get('C(children)[T.yes]', np.nan)
ols_children_p = ols_model.pvalues.get('C(children)[T.yes]', np.nan)

logit_children_coef = logit_model.params.get('C(children)[T.yes]', np.nan)
logit_children_p = logit_model.pvalues.get('C(children)[T.yes]', np.nan)

marginal_children = margeff.loc['C(children)[T.yes]'] if 'C(children)[T.yes]' in margeff.index else None

# Save results to a text file for inspection
with open('analysis_results.txt', 'w') as f:
    f.write('Descriptive summary by children:\n')
    f.write(summary.to_string())
    f.write('\n\n')
    f.write(f"Welch t-test (affairs yes vs no): stat={ttest.statistic:.4f}, p={ttest.pvalue:.4g}\n\n")
    f.write('OLS (affairs) with controls:\n')
    f.write(ols_model.summary().as_text())
    f.write('\n\n')
    f.write('Logit (any affair) with controls:\n')
    f.write(logit_model.summary().as_text())
    f.write('\n\n')
    f.write('Logit marginal effects (overall):\n')
    f.write(margeff.to_string())

print(summary)
print('\nWelch t-test:', ttest)
print('\nOLS children coef:', ols_children_coef, 'p=', ols_children_p)
print('Logit children coef:', logit_children_coef, 'p=', logit_children_p)
if marginal_children is not None:
    print('\nMarginal effect (children yes):')
    print(marginal_children)
