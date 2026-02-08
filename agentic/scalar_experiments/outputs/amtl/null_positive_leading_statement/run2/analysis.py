import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm

# Load data
_df = pd.read_csv('amtl.csv')

# Prepare variables
_df = _df.copy()
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']

# Fit binomial GLM with logit link using proportion with trial weights
model = smf.glm(
    formula='amtl_rate ~ is_human + age + prob_male + C(tooth_class)',
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['sockets'],
)
result = model.fit()

coef = float(result.params['is_human'])
se = float(result.bse['is_human'])
if se == 0:
    z = 0.0
else:
    z = coef / se

# Convert z to a 0..1 confidence scale, then to -100..100
conf = max(0.0, min(1.0, 2 * norm.cdf(abs(z)) - 1))
score = int(round(np.sign(coef) * conf * 100))

# Ensure within [-100, 100]
score = max(-100, min(100, score))

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(str(score))

# Also store a minimal analysis summary for audit/debugging
summary = {
    'coef_is_human': coef,
    'se_is_human': se,
    'z_is_human': z,
    'conf': conf,
    'score': score,
    'n_rows': int(len(_df)),
}
with open('analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
