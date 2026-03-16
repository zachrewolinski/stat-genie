import pandas as pd

_df = pd.read_csv('mortgage.csv')

cols = ['accept','deny','self_employed','denied_PMI','consumer_credit','PI_ratio','loan_to_value','female']
for c in cols:
    if c in _df.columns:
        print(c, _df[c].value_counts(dropna=False).head())

# pairwise equality
for a in ['accept','deny','self_employed']:
    for b in ['accept','deny','self_employed']:
        if a>=b: continue
        if a in _df.columns and b in _df.columns:
            eq = (_df[a] == _df[b]).mean()
            print(f'equal proportion {a} vs {b}: {eq:.3f}')
            print(pd.crosstab(_df[a], _df[b], dropna=False))

# Try to see which column looks like denial (rare=1) or acceptance (rare=1) by correlation with deny/accept names
# If there is a typical dataset, denial rates around 10-20%. So column with mean ~0.1 might be deny.

print('\nmeans:')
for c in ['accept','deny','self_employed','denied_PMI','consumer_credit','PI_ratio','loan_to_value','female']:
    if c in _df.columns:
        print(c, _df[c].mean())
