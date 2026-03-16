import pandas as pd

csv_path='teachingratings.csv'
df=pd.read_csv(csv_path)
cat_cols=[c for c in df.columns if df[c].dtype=='object']
print('categorical', cat_cols)
for c in cat_cols:
    print(c, df[c].unique()[:10], df[c].value_counts())

# numeric summary
num_cols=[c for c in df.columns if df[c].dtype!='object']
print('numeric', num_cols)
print(df[num_cols].describe())
