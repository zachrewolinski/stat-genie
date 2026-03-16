import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

print('Columns:', df.columns.tolist())
print('Head:')
print(df.head())

# basic info for binary columns
binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
print('Binary cols:', binary_cols)

# check relationships between accept/deny
if 'deny' in df.columns and 'accept' in df.columns:
    print('deny/accept crosstab:')
    print(pd.crosstab(df['deny'], df['accept']))
    print('mean deny', df['deny'].mean(), 'mean accept', df['accept'].mean())
    print('share accept == 1 - deny', (df['accept'] == (1-df['deny'])).mean())

# If accept exists, define approval as accept else as 1-deny
if 'accept' in df.columns:
    approval = df['accept']
    print('Using accept as approval')
elif 'deny' in df.columns:
    approval = 1 - df['deny']
    print('Using 1-deny as approval')
else:
    approval = None

if approval is not None:
    df = df.copy()
    df['approval'] = approval

# Identify gender column: female maybe
if 'female' in df.columns:
    print('female value counts:')
    print(df['female'].value_counts(dropna=False))

# Simple difference in approval rate by gender
if approval is not None and 'female' in df.columns:
    rates = df.groupby('female')['approval'].mean()
    counts = df['female'].value_counts()
    print('Approval rates by female:')
    print(rates)
    print('Counts:')
    print(counts)
    # two-proportion z-test
    from statsmodels.stats.proportion import proportions_ztest
    succ = df.groupby('female')['approval'].sum().reindex([0,1]).fillna(0)
    n = df['female'].value_counts().reindex([0,1]).fillna(0)
    stat, pval = proportions_ztest(succ, n)
    print('Two-proportion z-test stat', stat, 'p', pval)

# Logistic regression: approval ~ female + controls
# Choose controls among numeric columns excluding target-like and identifiers
controls = []
exclude = {'approval', 'accept', 'deny', 'female'}
for c in df.columns:
    if c in exclude:
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        # skip columns that are obviously binary maybe? but include as controls if binary too.
        controls.append(c)

print('Controls:', controls)

if approval is not None and 'female' in df.columns:
    # Build formula
    # Avoid perfect multicollinearity: drop one of accept/deny is already excluded
    # Also drop Unnamed: 0 if looks like index
    if 'Unnamed: 0' in controls:
        controls.remove('Unnamed: 0')
    formula = 'approval ~ female'
    if controls:
        formula += ' + ' + ' + '.join(controls)
    print('Formula:', formula)
    # Fit logistic regression
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    print(model.summary())
    # Extract female coefficient
    if 'female' in model.params:
        coef = model.params['female']
        se = model.bse['female']
        pval = model.pvalues['female']
        odds_ratio = np.exp(coef)
        print('female coef', coef, 'se', se, 'p', pval, 'odds ratio', odds_ratio)
