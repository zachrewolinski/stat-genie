import pandas as pd
import numpy as np
from scipy.special import expit

df = pd.read_csv('amtl.csv')
prod = df['num_amtl'] * df['sockets']
close_int = np.isclose(prod, np.round(prod), atol=1e-6)
print('prod close to int fraction', close_int.mean())
print('prod min/max', prod.min(), prod.max())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('prod head', prod.head(10).tolist())

p = expit(df['num_amtl'])
print('expit range', p.min(), p.max())
print('expit* sockets range', (p*df['sockets']).min(), (p*df['sockets']).max())
print('expit* sockets close int fraction', np.isclose(p*df['sockets'], np.round(p*df['sockets']), atol=0.1).mean())

pp = np.sin(df['num_amtl'])**2
print('sin^2 range', pp.min(), pp.max())
print('sin^2*sockets close int fraction', np.isclose(pp*df['sockets'], np.round(pp*df['sockets']), atol=0.1).mean())
