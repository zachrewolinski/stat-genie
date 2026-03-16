import pandas as pd


df = pd.read_csv('reading.csv')

df['derived_speed_wpm'] = df['feature7'] / (df['feature5'] / 60000.0)

for col in ['feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature19']:
    corr = df[[col,'feature20']].corr().iloc[0,1]
    print(f'corr(feature20, {col})= {corr}')

print('corr(feature20, derived_speed_wpm)=', df[['feature20','derived_speed_wpm']].corr().iloc[0,1])

print('feature4 ms quantiles', df['feature4'].quantile([0.1,0.5,0.9]).to_dict())
print('feature5 ms quantiles', df['feature5'].quantile([0.1,0.5,0.9]).to_dict())
print('feature20 quantiles', df['feature20'].quantile([0.1,0.5,0.9]).to_dict())
