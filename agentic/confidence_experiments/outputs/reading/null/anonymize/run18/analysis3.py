import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
for col in ['feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature19']:
    print(col, df[col].corr(df['feature20']))

# Maybe feature20 is reading speed (words per minute) if feature7 / feature4? check correlation with inverse time
inv_time = 1/(df['feature5']+1)  # avoid zero
print('corr feature20 with inverse feature5:', df['feature20'].corr(inv_time))
inv_time4 = 1/(df['feature4']+1)
print('corr feature20 with inverse feature4:', df['feature20'].corr(inv_time4))

# check correlation with feature7/feature20 etc
print('corr feature5 with feature7/feature20', df['feature5'].corr(df['feature7']/df['feature20']))

# compute derived reading speed per minute based on feature20 maybe? maybe feature20 is time per word? check if feature5/feature7 correlates
ratio = df['feature5'] / df['feature7']
print('ratio feature5/feature7 stats', ratio.describe())
print('corr feature20 with feature5/feature7', df['feature20'].corr(ratio))
