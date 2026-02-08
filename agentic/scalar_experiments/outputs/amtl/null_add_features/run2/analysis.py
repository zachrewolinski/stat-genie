import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')
_df = _df[['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']].copy()
_df = _df[_df['sockets'] > 0].copy()
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)
_df['log_sockets'] = np.log(_df['sockets'])

# Poisson GLM with offset for exposure (sockets)
model = smf.glm(
    'num_amtl ~ is_human + age + prob_male + C(tooth_class)',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_sockets']
)
res = model.fit(cov_type='HC0')

# Effect size for humans vs non-humans
coef = res.params['is_human']
se = res.bse['is_human']
rr = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Average predicted rate per socket difference
_df_h = _df.copy(); _df_h['is_human'] = 1
_df_n = _df.copy(); _df_n['is_human'] = 0
pred_h = res.predict(_df_h, offset=_df_h['log_sockets'])
pred_n = res.predict(_df_n, offset=_df_n['log_sockets'])
rate_h = float((pred_h / _df_h['sockets']).mean())
rate_n = float((pred_n / _df_n['sockets']).mean())
rate_diff = rate_h - rate_n

# Heuristic Likert mapping for direction and strength
p = float(res.pvalues['is_human'])

def likert(rr_val, p_val, diff_val):
    # Direction: positive means humans higher, negative means lower
    direction = 1 if rr_val > 1 else -1
    # Magnitude buckets from tiny to large
    effect = abs(np.log(rr_val))
    if effect >= 0.3 or abs(diff_val) >= 0.05:
        base = 70
    elif effect >= 0.2 or abs(diff_val) >= 0.02:
        base = 50
    elif effect >= 0.1 or abs(diff_val) >= 0.01:
        base = 30
    elif effect >= 0.05 or abs(diff_val) >= 0.005:
        base = 15
    else:
        base = 10

    # Downweight if not statistically strong
    if p_val < 0.001:
        weight = 1.0
    elif p_val < 0.01:
        weight = 0.8
    elif p_val < 0.05:
        weight = 0.6
    elif p_val < 0.1:
        weight = 0.4
    else:
        weight = 0.25

    score = int(round(direction * base * weight))
    return max(-100, min(100, score))

score = likert(rr, p, rate_diff)

print('is_human coef:', coef)
print('rate ratio:', rr, 'CI:', (ci_low, ci_high), 'p:', p)
print('rate_human:', rate_h, 'rate_nonhuman:', rate_n, 'diff:', rate_diff)
print('likert_score:', score)
