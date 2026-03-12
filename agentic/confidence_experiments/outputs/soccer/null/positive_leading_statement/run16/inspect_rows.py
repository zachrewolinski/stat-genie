import sys, os
for p in ["", os.getcwd()]:
    if p in sys.path:
        sys.path.remove(p)
import pandas as pd

df = pd.read_csv('soccer.csv')

skin = df[['rater1','rater2']].mean(axis=1)

bins = [0,0.25,0.5,0.75,1.0]
labels = ['<=0.25','0.25-0.5','0.5-0.75','0.75-1']
skin_non = skin.dropna()

binned = pd.cut(skin_non, bins=bins, include_lowest=True, right=True, labels=labels)
print(binned.value_counts().sort_index())

print('skin mean', skin_non.mean())
print('skin median', skin_non.median())
