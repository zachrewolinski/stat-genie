import pandas as pd
_df = pd.read_csv('mortgage.csv')
print('corr accept-self_employed', _df['accept'].corr(_df['self_employed']))
print('crosstab accept vs self_employed:')
print(pd.crosstab(_df['accept'], _df['self_employed']))

print('corr accept-denied_PMI', _df['accept'].corr(_df['denied_PMI']))
print('corr accept-consumer_credit', _df['accept'].corr(_df['consumer_credit']))
