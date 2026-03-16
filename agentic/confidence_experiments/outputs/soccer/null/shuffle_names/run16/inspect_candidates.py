import pandas as pd

path='soccer.csv'
df=pd.read_csv(path)

# candidate games column: redCards (1-47)

candidates=['yellowCards','meanExp','yellowReds']

for c in candidates:
    exceeds=(df[c] > df['redCards']).sum()
    print(c,'exceeds games count',exceeds)

# check yellowReds and meanExp etc

