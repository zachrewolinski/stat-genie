import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load data
raw = pd.read_csv('amtl.csv')

# Map columns to intended variables based on metadata
# sockets -> tooth_class
# prob_male -> specimen_id (unused)
# genus -> num_amtl (count missing)
# age -> num_observable (count observed sockets)
# pop -> age_at_death
# num_amtl -> stdev_age (age uncertainty)
# stdev_age -> prob_male (sex estimate)
# tooth_class -> genus (species)
# specimen -> population/region

df = pd.DataFrame({
    'tooth_class': raw['sockets'],
    'specimen_id': raw['prob_male'],
    'num_amtl': raw['genus'],
    'num_observable': raw['age'],
    'age_at_death': raw['pop'],
    'stdev_age': raw['num_amtl'],
    'prob_male': raw['stdev_age'],
    'genus': raw['tooth_class'],
    'population': raw['specimen'],
})

# Drop rows with impossible counts
mask = df['num_amtl'] <= df['num_observable']
df_clean = df[mask].copy()

# Avoid zero or missing obs
df_clean = df_clean[(df_clean['num_observable'] > 0) & df_clean['num_amtl'].notna()]

# Create proportion for GLM with binomial and weights
# Use Homo sapiens as reference by setting categorical ordering

df_clean['genus'] = pd.Categorical(df_clean['genus'], categories=['Homo sapiens','Pan','Papio','Pongo'])
df_clean['tooth_class'] = pd.Categorical(df_clean['tooth_class'], categories=['Anterior','Posterior','Premolar'])

# Fit binomial GLM
# Response as proportion with var_weights = num_observable

df_clean['amtl_rate'] = df_clean['num_amtl'] / df_clean['num_observable']

formula = 'amtl_rate ~ C(genus) + age_at_death + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=df_clean, family=sm.families.Binomial(), var_weights=df_clean['num_observable'])
res = model.fit()

# Extract coefficient for Homo sapiens vs others
params = res.params
conf = res.conf_int()

# For each non-human genus, compute log-odds difference relative to Homo sapiens (reference)
comparisons = {}
for g in ['Pan','Papio','Pongo']:
    key = f'C(genus)[T.{g}]'
    if key in params:
        # This is effect of g relative to Homo sapiens. We want Homo sapiens vs g, so negate.
        log_odds_hs_vs_g = -params[key]
        ci_low, ci_high = -conf.loc[key,1], -conf.loc[key,0]
        comparisons[g] = {
            'log_odds': log_odds_hs_vs_g,
            'odds_ratio': float(np.exp(log_odds_hs_vs_g)),
            'ci_low': float(np.exp(ci_low)),
            'ci_high': float(np.exp(ci_high)),
            'p_value': float(res.pvalues[key])
        }

# Overall model-predicted mean difference: predicted human rate minus mean non-human rate
# compute marginal predictions

pred = res.predict(df_clean)

df_clean['pred_rate'] = pred

human_mean = df_clean[df_clean['genus']=='Homo sapiens']['pred_rate'].mean()
nonhuman_mean = df_clean[df_clean['genus']!='Homo sapiens']['pred_rate'].mean()

print('rows total', len(df), 'rows used', len(df_clean))
print('human_pred_mean', human_mean)
print('nonhuman_pred_mean', nonhuman_mean)
print('diff', human_mean - nonhuman_mean)
print('comparisons', comparisons)

# Save results to csv for review
out = pd.DataFrame(comparisons).T
out.to_csv('model_comparisons.csv', index=True)

