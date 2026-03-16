import json
import pandas as pd
import numpy as np
from pathlib import Path

base = Path('.')
info = json.loads((base / 'info.json').read_text())
fields = info['data_desc']['fields']

# Build mapping: csv column name -> description (true meaning)
col_desc = {f['column']: f['properties']['description'] for f in fields}

# Load data
csv_path = base / 'soccer.csv'
_df = pd.read_csv(csv_path)

print('rows', _df.shape[0], 'cols', _df.shape[1])
print('columns', list(_df.columns))

# Show column stats for numeric columns
num_cols = _df.select_dtypes(include=[np.number]).columns
print('numeric columns', list(num_cols))

# show unique sample values for columns with limited unique
for col in _df.columns:
    if _df[col].nunique() <= 10:
        print(col, 'unique', sorted(_df[col].unique().tolist())[:10])

# display mapping for key descriptions
for col, desc in col_desc.items():
    if 'red card' in desc.lower() or 'skin rating' in desc.lower() or 'skin' in desc.lower() or 'games' in desc.lower():
        print(col, '->', desc)

# Compute mean skin rating from rater1 and rater2 (rater2 appears to be in column with desc 'Skin rating of photo by rater 2 ...')
# Find columns for rater1 and rater2 based on description
rater1_col = None
rater2_col = None
red_cards_col = None
games_col = None

for col, desc in col_desc.items():
    d = desc.lower()
    if 'skin rating of photo by rater 1' in d:
        rater1_col = col
    if 'skin rating of photo by rater 2' in d:
        rater2_col = col
    if 'number of red cards player received from referee' in d:
        red_cards_col = col
    if 'number of games in the player-referee dyad' in d:
        games_col = col

print('rater1_col', rater1_col, 'rater2_col', rater2_col, 'red_cards_col', red_cards_col, 'games_col', games_col)

# Basic summary for these
for col in [rater1_col, rater2_col, red_cards_col, games_col]:
    if col is None:
        continue
    print(col, 'describe', _df[col].describe())

