import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

path = 'mortgage.csv'
df = pd.read_csv(path)
print('columns', df.columns.tolist())
print(df.head())
print(df.describe(include='all'))

# identify outcome columns
for col in ['accept','deny']:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).head())

# check if accept and deny are complements
if 'accept' in df.columns and 'deny' in df.columns:
    comp = ((df['accept'] + df['deny']) == 1).mean()
    print('accept+deny==1 proportion', comp)

# outcome: approval? accept=1? verify by rate
if 'accept' in df.columns:
    # treat accept as approval
    outcome='accept'
elif 'deny' in df.columns:
    outcome='deny'
else:
    outcome=None
print('Outcome chosen', outcome)

# check female distribution
print('female counts', df['female'].value_counts(dropna=False))

# simple difference in approval rate by female
if outcome:
    rates = df.groupby('female')[outcome].mean()
    print('mean outcome by female', rates)
    # chi-square test
    tab = pd.crosstab(df['female'], df[outcome])
    print('crosstab', tab)
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    print('chi2', chi2, 'p', p, 'dof', dof)
    # two-proportion z-test
    # for outcome binary with female 0/1
    group1 = df[df['female']==1][outcome]
    group0 = df[df['female']==0][outcome]
    # handle counts
    n1, n0 = len(group1), len(group0)
    p1, p0 = group1.mean(), group0.mean()
    p_pool = (group1.sum()+group0.sum())/(n1+n0)
    z = (p1-p0)/np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n0))
    p_z = 2*(1-stats.norm.cdf(abs(z)))
    print('two-prop z', z, 'p', p_z, 'p1', p1, 'p0', p0)

# logistic regression controlling for other variables
# use all other numeric columns except outcome
if outcome:
    features = [c for c in df.columns if c not in [outcome] and c != 'Unnamed: 0']
    # drop rows with missing
    df_clean = df[features + [outcome]].dropna()
    # build formula
    # exclude outcome from features; ensure female included
    # For logistic, use outcome ~ female + others
    # Some columns might be binary but ok
    # Remove any non-numeric columns (none expected)
    # Check for collinearity: if accept and deny complement, include only features excluding deny or accept? 
    # If outcome is accept, exclude deny from features to avoid leakage.
    if outcome == 'accept' and 'deny' in features:
        features.remove('deny')
    if outcome == 'deny' and 'accept' in features:
        features.remove('accept')
    # also remove female from features then add separately
    if 'female' in features:
        features.remove('female')
    formula = outcome + ' ~ female'
    if features:
        formula += ' + ' + ' + '.join(features)
    print('formula', formula)
    model = smf.logit(formula, data=df_clean).fit(disp=False)
    print(model.summary())
    # odds ratio for female
    params = model.params
    conf = model.conf_int()
    or_female = np.exp(params['female'])
    ci = np.exp(conf.loc['female'])
    print('OR female', or_female, 'CI', ci.tolist())
