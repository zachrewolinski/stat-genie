import pandas as pd

df = pd.read_csv('affairs.csv')
for col in df.columns:
    vc = df[col].value_counts()
    max_count = vc.max()
    min_count = vc.min()
    print(col, 'unique', vc.size, 'max_count', max_count, 'min_count', min_count)

# check if education values are all unique
print('education all unique:', df['education'].is_unique)
