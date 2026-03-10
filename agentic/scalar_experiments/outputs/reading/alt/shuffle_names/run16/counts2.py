import pandas as pd

df = pd.read_csv('reading.csv')
for col in ['img_width','english_native','page_id','reader_view','scrolling_time']:
    print('\n', col)
    print(df[col].value_counts(dropna=False).head(10))
