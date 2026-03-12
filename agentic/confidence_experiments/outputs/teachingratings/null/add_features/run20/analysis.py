import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('teachingratings.csv')
print('columns', df.columns.tolist())
print('shape', df.shape)
print(df.head())

# ensure required columns
if 'beauty' not in df.columns or 'eval' not in df.columns:
    raise ValueError('Required columns not found')

# simple correlation
corr = df[['beauty', 'eval']].corr().iloc[0, 1]
print('corr beauty-eval', corr)

# basic OLS
model1 = smf.ols('eval ~ beauty', data=df).fit()
print(model1.summary())

# Build control variables if present
controls = []
for col in ['age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students', 'allstudents']:
    if col in df.columns:
        controls.append(col)

# create formula
if controls:
    # Use C() for categorical
    terms = []
    for c in controls:
        if df[c].dtype == 'object' or str(df[c].dtype).startswith('category'):
            terms.append(f'C({c})')
        else:
            terms.append(c)
    formula = 'eval ~ beauty + ' + ' + '.join(terms)
    model2 = smf.ols(formula, data=df).fit()
    print('controls formula', formula)
    print(model2.summary())
else:
    model2 = None

# standardized effect size (beta) for beauty in model2 if possible
if model2 is not None:
    zdf = df.copy()
    cont_cols = ['beauty'] + [c for c in controls if not (df[c].dtype == 'object' or str(df[c].dtype).startswith('category'))]
    for c in cont_cols:
        zdf[c] = (zdf[c] - zdf[c].mean()) / zdf[c].std(ddof=0)
    zformula = 'eval ~ beauty'
    for c in controls:
        if df[c].dtype == 'object' or str(df[c].dtype).startswith('category'):
            zformula += f' + C({c})'
        else:
            zformula += f' + {c}'
    zmodel = smf.ols(zformula, data=zdf).fit()
    print('standardized formula', zformula)
    print('std beta beauty', zmodel.params.get('beauty'))

