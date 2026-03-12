import json, math
with open('analysis_results.json','r') as f:
    data=json.load(f)
params=data['poisson_params']
print({k: math.exp(v) for k,v in params.items()})
