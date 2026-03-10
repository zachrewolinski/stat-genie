import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = 'teachingratings.csv'

df = pd.read_csv(DATA_PATH)
print('rows', len(df), 'cols', df.shape[1])
print('columns', list(df.columns))

# Basic info for target vars
for col in ['beauty','eval']:
    if col in df.columns:
        print(col, df[col].describe())

# Drop missing for main analysis
if 'beauty' not in df.columns or 'eval' not in df.columns:
    raise SystemExit('Required columns missing')

subset = df[['beauty','eval']].dropna()
print('n for beauty/eval', len(subset))

corr, corr_p = stats.pearsonr(subset['beauty'], subset['eval'])
print('pearson r', corr, 'p', corr_p)

# Simple OLS
X = sm.add_constant(subset['beauty'])
model_simple = sm.OLS(subset['eval'], X).fit(cov_type='HC3')
print('simple coef', model_simple.params['beauty'], 'p', model_simple.pvalues['beauty'])

# Controls if present
candidate_controls = ['age','gender','minority','division','native','tenure','students','credits','allstudents']
controls = [c for c in candidate_controls if c in df.columns]
print('controls', controls)

if controls:
    df_ctrl = df[['eval','beauty'] + controls].dropna()
    y = df_ctrl['eval']
    Xc = df_ctrl[['beauty'] + controls].copy()
    # Identify categorical controls (object or category)
    cat_cols = [c for c in controls if Xc[c].dtype == 'object' or str(Xc[c].dtype).startswith('category')]
    if cat_cols:
        Xc = pd.get_dummies(Xc, columns=cat_cols, drop_first=True)
    Xc = sm.add_constant(Xc)
    if 'prof' in df.columns:
        # cluster by professor if available (not in controls but maybe for se)
        clusters = df_ctrl.get('prof')
        if clusters is not None:
            model_ctrl = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': clusters})
        else:
            model_ctrl = sm.OLS(y, Xc).fit(cov_type='HC3')
    else:
        model_ctrl = sm.OLS(y, Xc).fit(cov_type='HC3')
    coef = model_ctrl.params['beauty']
    pval = model_ctrl.pvalues['beauty']
    print('control coef', coef, 'p', pval, 'n', len(df_ctrl))

    # Standardized effect (per 1 SD beauty)
    sd_beauty = df_ctrl['beauty'].std()
    effect_sd = coef * sd_beauty
    print('effect per 1 sd beauty', effect_sd)

# Also compute standardized coef from simple model
sd_b = subset['beauty'].std()
print('simple effect per 1 sd beauty', model_simple.params['beauty'] * sd_b)
