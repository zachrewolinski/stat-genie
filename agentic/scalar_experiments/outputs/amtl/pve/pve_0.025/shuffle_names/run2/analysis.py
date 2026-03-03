import pandas as pd

_df = pd.read_csv('amtl.csv')
print(_df.head())
print(_df.dtypes)
# check per specimen variability
specimen_col = 'prob_male'
print('num specimens', _df[specimen_col].nunique())
# compute number of rows per specimen
print('rows per specimen (min/max)', _df.groupby(specimen_col).size().min(), _df.groupby(specimen_col).size().max())
# columns that are constant within specimen
const_cols = []
for col in _df.columns:
    if col==specimen_col: continue
    nun = _df.groupby(specimen_col)[col].nunique()
    if (nun==1).all():
        const_cols.append(col)
print('constant within specimen:', const_cols)
# summary stats
print(_df.describe(include='all'))
