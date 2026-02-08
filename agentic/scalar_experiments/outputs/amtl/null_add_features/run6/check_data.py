import pandas as pd
DF = pd.read_csv('amtl.csv')
relevant_genera = {"Homo sapiens", "Pan", "Pongo", "Papio"}
DF = DF[DF['genus'].isin(relevant_genera)].copy()
DF = DF.dropna(subset=['num_amtl','sockets','age','prob_male','tooth_class','genus'])
DF = DF[DF['sockets']>0]
print('rows', len(DF))
print('num_amtl> sockets', (DF['num_amtl']>DF['sockets']).sum())
print('num_amtl<0', (DF['num_amtl']<0).sum())
print('sockets min', DF['sockets'].min())
print('num_amtl min/max', DF['num_amtl'].min(), DF['num_amtl'].max())
print('any nan proportion', (DF['num_amtl']/DF['sockets']).isna().sum())
print('proportion min/max', (DF['num_amtl']/DF['sockets']).min(), (DF['num_amtl']/DF['sockets']).max())
