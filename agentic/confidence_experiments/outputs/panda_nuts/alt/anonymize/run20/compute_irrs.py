import numpy as np
coef = {'age':0.135295,'sex_m':1.274674,'help_y':-0.724406}
for k,v in coef.items():
    print(k, np.exp(v))
