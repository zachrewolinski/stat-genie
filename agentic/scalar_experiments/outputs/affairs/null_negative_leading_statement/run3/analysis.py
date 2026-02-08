import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic preprocessing
# Ensure children is categorical with 'no' as reference
if df['children'].dtype != 'O':
    df['children'] = df['children'].astype(str)
# Normalize values
df['children'] = df['children'].str.strip().str.lower()
df['gender'] = df['gender'].str.strip().str.lower()

# Create indicator for any affair
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Group stats
grp = df.groupby('children')
summary = grp['affairs'].agg(['mean', 'median', 'count', 'std'])
any_rate = grp['any_affair'].mean()

print('Group affairs summary by children:')
print(summary)
print('\nAny-affair rate by children:')
print(any_rate)

# Simple difference in means
children_yes = df[df['children'] == 'yes']['affairs']
children_no = df[df['children'] == 'no']['affairs']
diff_mean = children_yes.mean() - children_no.mean()
print(f"\nMean difference (yes - no): {diff_mean:.4f}")

# OLS on affairs count (not ideal but baseline)
ols = smf.ols('affairs ~ C(children)', data=df).fit()
print('\\nOLS affairs ~ children')
print(ols.summary().tables[1])

# Poisson regression (count model)
poisson = smf.glm(
    'affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=df,
    family=sm.families.Poisson(),
).fit()
print('\\nPoisson regression with controls')
print(poisson.summary().tables[1])

# Negative binomial regression (if available)
try:
    nb = smf.glm(
        'affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
        data=df,
        family=sm.families.NegativeBinomial(),
    ).fit()
    print('\\nNegative binomial regression with controls')
    print(nb.summary().tables[1])
except Exception as e:
    print('Negative binomial failed:', e)

# Logistic regression for any affair
logit = smf.logit(
    'any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=df,
).fit(disp=False)
print('\\nLogistic regression for any affair')
print(logit.summary().tables[1])

# Store key coefficients for later
coef = {
    'ols_children': ols.params.get('C(children)[T.yes]', np.nan),
    'ols_p': ols.pvalues.get('C(children)[T.yes]', np.nan),
    'poisson_children': poisson.params.get('C(children)[T.yes]', np.nan),
    'poisson_p': poisson.pvalues.get('C(children)[T.yes]', np.nan),
    'logit_children': logit.params.get('C(children)[T.yes]', np.nan),
    'logit_p': logit.pvalues.get('C(children)[T.yes]', np.nan),
}
print('\\nKey coefficients:')
for k, v in coef.items():
    print(f"{k}: {v}")
