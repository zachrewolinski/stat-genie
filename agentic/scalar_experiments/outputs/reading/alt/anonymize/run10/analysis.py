import pandas as pd
import numpy as np

_df = pd.read_csv('reading.csv')
print(_df.head())
print(_df.describe(include='all').transpose().head(30))
print(_df.dtypes)

# compute candidate speeds
for tcol in ['feature4','feature5']:
    speed_wpm = _df['feature7'] / (_df[tcol] / 60000.0)
    print(tcol, speed_wpm.describe())

# check feature20 range
print('feature20 describe', _df['feature20'].describe())
# correlation between feature20 and computed speeds
for tcol in ['feature4','feature5']:
    speed_wpm = _df['feature7'] / (_df[tcol] / 60000.0)
    corr = np.corrcoef(speed_wpm, _df['feature20'])[0,1]
    print('corr feature20 with speed from', tcol, corr)

# correlation of feature20 with time variables
for col in ['feature4','feature5','feature6','feature7']:
    corr = np.corrcoef(_df[col], _df['feature20'])[0,1]
    print('corr feature20 with', col, corr)

# check typical scales to interpret feature20
print('feature20 quantiles', _df['feature20'].quantile([0.01,0.1,0.5,0.9,0.99]))
