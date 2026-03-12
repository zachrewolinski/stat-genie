import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')
print(_df[['num_amtl','sockets']].corr())
