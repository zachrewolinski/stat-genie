import pandas as pd
_df = pd.read_csv('boxes.csv')
_df['majority_choice'] = (_df['majority_first'] == 2).astype(int)
bins = [3.5,5.5,7.5,9.5,11.5,13.5,15.5]
labels = ['4-5','6-7','8-9','10-11','12-13','14-15']
_df['age_group'] = pd.cut(_df['age'], bins=bins, labels=labels)
print(_df['age_group'].value_counts().sort_index())
