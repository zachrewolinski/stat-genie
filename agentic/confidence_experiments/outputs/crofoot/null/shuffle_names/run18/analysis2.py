import pandas as pd

df=pd.read_csv('crofoot.csv')
# check possible relationships
pairs=[('f_other','focal','other'),('win','focal','other'),('f_other','dist_focal','f_focal'),('win','dist_focal','f_focal'),('f_other','dist_focal','other'),('win','dist_focal','other'),('f_other','focal','f_focal'),('win','focal','f_focal')]
for total, a, b in pairs:
    matches=(df[total]==df[a]+df[b]).mean()
    print(f'{total}=={a}+{b}: {matches:.2f}')
