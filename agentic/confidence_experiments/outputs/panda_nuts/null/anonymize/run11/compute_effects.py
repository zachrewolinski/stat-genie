import json
import math

with open('analysis_results.json') as f:
    res = json.load(f)

nb = res['negative_binomial']

params = nb['params']
conf = nb['conf_int']

for key in ['feature2', 'C(feature3)[T.m]', 'C(feature7)[T.y]']:
    beta = params[key]
    lo = conf[key]['low']
    hi = conf[key]['high']
    irr = math.exp(beta)
    irr_lo = math.exp(lo)
    irr_hi = math.exp(hi)
    print(key, beta, irr, irr_lo, irr_hi, nb['pvalues'][key])
