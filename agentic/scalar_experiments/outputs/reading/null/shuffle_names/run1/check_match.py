import pandas as pd

df = pd.read_csv('reading.csv')
# compare correct_rate to device>0
match = (df['correct_rate'] == (df['device'] > 0).astype(float))
print('match proportion device>0 vs correct_rate:', match.mean())
# compare correct_rate to dyslexia>0
match2 = (df['correct_rate'] == (df['dyslexia'] > 0).astype(float))
print('match proportion dyslexia>0 vs correct_rate:', match2.mean())

# check if correct_rate equals dyslexia_bin
match3 = (df['correct_rate'] == df['dyslexia_bin']).mean()
print('match proportion correct_rate vs dyslexia_bin:', match3)
