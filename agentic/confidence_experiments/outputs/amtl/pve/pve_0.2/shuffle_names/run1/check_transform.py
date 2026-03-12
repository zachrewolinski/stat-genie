import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
exp_genus = np.exp(df['genus'])
# count how close to integer
close_to_int = np.isclose(exp_genus, np.round(exp_genus), atol=0.01)
print('exp(genus) min/max', exp_genus.min(), exp_genus.max())
print('exp(genus) close to int fraction', close_to_int.mean())
# try log1p inverse: maybe genus = log(num_missing)??
log1p_inv = np.expm1(df['genus'])
close_to_int2 = np.isclose(log1p_inv, np.round(log1p_inv), atol=0.01)
print('expm1(genus) min/max', log1p_inv.min(), log1p_inv.max())
print('expm1 close to int fraction', close_to_int2.mean())
