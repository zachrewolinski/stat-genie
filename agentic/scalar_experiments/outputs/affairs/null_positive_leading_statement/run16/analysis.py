import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic cleaning
# children is categorical yes/no
_df['children'] = _df['children'].astype('category')

# Create binary outcome: any affairs (>0)
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Group stats
summary = _df.groupby('children')['affairs'].agg(['count', 'mean', 'median']).rename_axis('children')
prop_any = _df.groupby('children')['any_affair'].mean().rename('prop_any')

# Difference in means (yes - no)
mean_yes = summary.loc['yes', 'mean']
mean_no = summary.loc['no', 'mean']
mean_diff = mean_yes - mean_no

# Difference in proportion any affair
prop_yes = prop_any.loc['yes']
prop_no = prop_any.loc['no']
prop_diff = prop_yes - prop_no

# Regression: OLS on affairs count with controls
# Controls: gender, age, yearsmarried, religiousness, education, occupation, rating
# children as categorical
ols_model = smf.ols(
    'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=_df
).fit()

# Logistic regression for any_affair
logit_model = smf.logit(
    'any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=_df
).fit(disp=False)

# Extract child effect: for C(children)[T.yes]
ols_coef = ols_model.params.get('C(children)[T.yes]', np.nan)
ols_p = ols_model.pvalues.get('C(children)[T.yes]', np.nan)

logit_coef = logit_model.params.get('C(children)[T.yes]', np.nan)
logit_p = logit_model.pvalues.get('C(children)[T.yes]', np.nan)

# Convert logit coef to odds ratio
odds_ratio = np.exp(logit_coef)

# Simple heuristic to map results to Likert scalar
# Negative coef/diffs indicate fewer affairs with children.
# Use magnitude and significance to scale from -100 to 100.
score = 0

# Direction from raw differences
if mean_diff < 0:
    score -= 15
elif mean_diff > 0:
    score += 15

if prop_diff < 0:
    score -= 15
elif prop_diff > 0:
    score += 15

# Add weight for regression effects
if ols_coef < 0:
    score -= 20
elif ols_coef > 0:
    score += 20

if logit_coef < 0:
    score -= 20
elif logit_coef > 0:
    score += 20

# Significance boosts
for p in [ols_p, logit_p]:
    if p < 0.001:
        score -= 20
    elif p < 0.01:
        score -= 15
    elif p < 0.05:
        score -= 10
    elif p < 0.1:
        score -= 5

# Clamp
score = int(max(-100, min(100, score)))

# Print summary for inspection
print('Summary by children:\n', summary)
print('\nProportion any affair:\n', prop_any)
print('\nMean diff (yes - no):', mean_diff)
print('Prop diff (yes - no):', prop_diff)
print('\nOLS child coef:', ols_coef, 'p=', ols_p)
print('Logit child coef:', logit_coef, 'odds ratio=', odds_ratio, 'p=', logit_p)
print('\nLikert score:', score)

# Write conclusion
with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(str(score))
