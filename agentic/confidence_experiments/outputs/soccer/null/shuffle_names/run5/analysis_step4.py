import pandas as pd

_df = pd.read_csv('soccer.csv')
for col in ['rater1','nExp','meanExp','yellowCards','yellowReds']:
    print(col, sorted(_df[col].unique())[:10], '... last', sorted(_df[col].unique())[-10:])
    print(col, 'unique count', _df[col].nunique())

