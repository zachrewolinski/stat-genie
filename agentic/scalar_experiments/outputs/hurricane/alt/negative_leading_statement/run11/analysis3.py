import pandas as pd
import numpy as np

_df = pd.read_csv('hurricane.csv')
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# group by gender_mf
summary = _df.groupby('gender_mf').agg(
    n=('alldeaths','size'),
    deaths_mean=('alldeaths','mean'),
    deaths_median=('alldeaths','median'),
    log_deaths_mean=('log_deaths','mean'),
    masfem_mean=('masfem','mean'),
)

summary.to_csv('gender_summary.csv')
print(summary)
