import pandas as pd

df = pd.read_csv('teachingratings.csv')
print(df[['beauty', 'allstudents']].describe().T)
