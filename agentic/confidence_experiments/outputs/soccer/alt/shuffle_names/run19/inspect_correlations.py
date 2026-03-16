import pandas as pd


df = pd.read_csv('soccer.csv')

# assume games column is 'redCards'
# candidate card columns
candidates = ['defeats','rater2','player','yellowReds','meanExp','yellowCards']
for col in candidates:
    corr = df['redCards'].corr(df[col])
    print(col, 'corr_with_redCards', corr)
