import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

path = 'mortgage.csv'

df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))

# Check unique values for key columns
for col in ['female', 'deny', 'accept']:
    if col in df.columns:
        print(col, df[col].value_counts().sort_index())

# Check if accept and deny are complements
if set(['accept','deny']).issubset(df.columns):
    print('accept + deny unique values:', (df['accept'] + df['deny']).unique())
    print('corr accept deny:', df['accept'].corr(df['deny']))

# Crosstab female vs deny/accept
if 'female' in df.columns:
    if 'deny' in df.columns:
        ct = pd.crosstab(df['female'], df['deny'], normalize='index')
        print('\nCrosstab female vs deny (row proportions):')
        print(ct)
    if 'accept' in df.columns:
        ct2 = pd.crosstab(df['female'], df['accept'], normalize='index')
        print('\nCrosstab female vs accept (row proportions):')
        print(ct2)

# Logistic regression: predict deny with female only and with controls
# Identify numeric controls (exclude target and female)
if set(['deny','female']).issubset(df.columns):
    y = df['deny']
    X_cols = [c for c in df.columns if c not in ['deny','accept']]
    # Build formula with female + other numeric columns
    # Exclude non-numeric if any
    numeric_cols = [c for c in X_cols if pd.api.types.is_numeric_dtype(df[c])]
    # Ensure female included
    if 'female' not in numeric_cols:
        numeric_cols.append('female')
    # Remove duplicate
    numeric_cols = list(dict.fromkeys(numeric_cols))
    # Simple model
    try:
        m1 = logit('deny ~ female', data=df).fit(disp=0)
        print('\nLogit deny ~ female')
        print(m1.summary())
    except Exception as e:
        print('logit deny ~ female failed', e)
    # Full model with controls
    # Remove target and possibly col with all unique (Unnamed: 0 index)
    numeric_cols = [c for c in numeric_cols if c not in ['deny','accept']]
    # exclude columns with too many unique values? keep
    # Build formula
    formula = 'deny ~ ' + ' + '.join(numeric_cols)
    try:
        m2 = logit(formula, data=df).fit(disp=0, maxiter=200)
        print('\nLogit deny ~ controls')
        print(m2.summary())
    except Exception as e:
        print('logit deny ~ controls failed', e)

# Logistic regression: accept as outcome
if set(['accept','female']).issubset(df.columns):
    try:
        m3 = logit('accept ~ female', data=df).fit(disp=0)
        print('\nLogit accept ~ female')
        print(m3.summary())
    except Exception as e:
        print('logit accept ~ female failed', e)
    X_cols2 = [c for c in df.columns if c not in ['accept','deny']]
    numeric_cols2 = [c for c in X_cols2 if pd.api.types.is_numeric_dtype(df[c])]
    if 'female' not in numeric_cols2:
        numeric_cols2.append('female')
    numeric_cols2 = list(dict.fromkeys(numeric_cols2))
    formula2 = 'accept ~ ' + ' + '.join(numeric_cols2)
    try:
        m4 = logit(formula2, data=df).fit(disp=0, maxiter=200)
        print('\nLogit accept ~ controls')
        print(m4.summary())
    except Exception as e:
        print('logit accept ~ controls failed', e)
