import json
import math

with open('analysis_output.json') as f:
    data=json.load(f)

nb = data['negbin_cluster']
po = data['poisson_cluster']

for name, model in [('negbin', nb), ('poisson', po)]:
    age = model['age']['coef']
    sex = model['C(sex)[T.m]']['coef']
    helpc = model['C(help)[T.y]']['coef']
    print(name)
    print('age rr', math.exp(age))
    print('sex rr', math.exp(sex))
    print('help rr', math.exp(helpc))
