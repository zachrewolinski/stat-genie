import pandas as pd

df = pd.read_csv('reading.csv')
cat_cols = ['speed','scrolling_time','english_native','page_id','reader_view','img_width']
for col in cat_cols:
    vals = df[col].dropna().unique()
    print(col, 'nunique', len(vals), 'sample', vals[:10])

for col in ['device','dyslexia','dyslexia_bin','correct_rate']:
    vals = sorted(df[col].dropna().unique())
    print(col, vals)
