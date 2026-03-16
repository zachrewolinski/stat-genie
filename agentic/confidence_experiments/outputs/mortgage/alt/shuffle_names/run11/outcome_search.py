import pandas as pd

cont_cols = ['mortgage_credit','housing_expense_ratio','Unnamed: 0']

df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]

for b in binary_cols:
    print('\n', b)
    for c in cont_cols:
        mean1 = df.loc[df[b]==1, c].mean()
        mean0 = df.loc[df[b]==0, c].mean()
        print(f'  {c}: mean1 {mean1:.3f} mean0 {mean0:.3f} diff {mean1-mean0:.3f}')
