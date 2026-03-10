import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
df = pd.read_csv('reading.csv')
# compute wpm based on feature7 words and feature5 reading time (ms)
df['wpm_calc'] = df['feature7'] / (df['feature5'] / 1000.0 / 60.0)
# alternative using feature4 total time
_df = df.copy()
_df['wpm_total'] = _df['feature7'] / (_df['feature4'] / 1000.0 / 60.0)
# correlation with feature20
print('corr with feature20 (wpm_calc):', df['wpm_calc'].corr(df['feature20']))
print('corr with feature20 (wpm_total):', _df['wpm_total'].corr(df['feature20']))
print('wpm_calc summary', df['wpm_calc'].describe())
print('wpm_total summary', _df['wpm_total'].describe())
print('feature20 summary', df['feature20'].describe())
# ratio feature20 to wpm_calc
ratio = df['feature20'] / df['wpm_calc']
print('ratio summary', ratio.describe())
