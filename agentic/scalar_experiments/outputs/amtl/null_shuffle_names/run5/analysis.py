import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

raw = pd.read_csv('amtl.csv')

# Map shuffled columns to semantic names based on value patterns and info.json descriptions
# sockets -> tooth_class (Anterior/Posterior/Premolar)
# prob_male -> specimen_id
# genus -> num_amtl (integer counts 0-12)
# age -> num_sockets (integer 2-14)
# pop -> age_at_death (years)
# num_amtl -> age_stdev (uncertainty of age at death)
# stdev_age -> prob_male (0-1)
# tooth_class -> genus (Homo sapiens/Pan/Papio/Pongo)
# specimen -> population/region

df = pd.DataFrame({
    'tooth_class': raw['sockets'],
    'specimen_id': raw['prob_male'],
    'num_amtl': raw['genus'].astype(int),
    'num_sockets': raw['age'].astype(int),
    'age_at_death': raw['pop'].astype(float),
    'age_stdev': raw['num_amtl'].astype(float),
    'prob_male': raw['stdev_age'].astype(float),
    'genus': raw['tooth_class'],
    'population': raw['specimen'],
})

# Remove rows where AMTL count exceeds sockets (data inconsistencies)
model_df = df[df['num_amtl'] <= df['num_sockets']].copy()
model_df = model_df[model_df['num_sockets'] > 0].copy()
model_df['amtl_rate'] = model_df['num_amtl'] / model_df['num_sockets']

# Fit binomial GLM with frequency weights (num_sockets)
formula = 'amtl_rate ~ C(genus) + age_at_death + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial(), freq_weights=model_df['num_sockets']).fit()

print(model.summary())

# Compute adjusted predicted probabilities for Homo sapiens vs non-human
non_human = ['Pan', 'Pongo', 'Papio']

# Average predicted prob for Homo sapiens (counterfactual, set genus to Homo for all rows)
hs_df = model_df.copy()
hs_df['genus'] = 'Homo sapiens'
hs_pred = model.predict(hs_df)

# Average predicted prob for non-humans using observed non-human rows
nh_df = model_df[model_df['genus'].isin(non_human)].copy()
nh_pred = model.predict(nh_df)

# Counterfactual equal mix of non-human genera for all rows
nh_counter = []
for g in non_human:
    tmp = model_df.copy()
    tmp['genus'] = g
    nh_counter.append(model.predict(tmp))
nh_pred_equal = np.mean(np.vstack(nh_counter), axis=0)

print("Adjusted mean predicted AMTL rate (Homo sapiens):", hs_pred.mean())
print("Adjusted mean predicted AMTL rate (Non-human observed):", nh_pred.mean())
print("Adjusted mean predicted AMTL rate (Non-human equal mix):", nh_pred_equal.mean())
print("Difference (Homo sapiens - Non-human observed):", hs_pred.mean() - nh_pred.mean())
print("Difference (Homo sapiens - Non-human equal mix):", hs_pred.mean() - nh_pred_equal.mean())

# Save summary numbers for later use
summary = {
    'n_total': len(df),
    'n_used': len(model_df),
    'hs_mean': float(hs_pred.mean()),
    'nh_obs_mean': float(nh_pred.mean()),
    'nh_equal_mean': float(nh_pred_equal.mean()),
    'diff_obs': float(hs_pred.mean() - nh_pred.mean()),
    'diff_equal': float(hs_pred.mean() - nh_pred_equal.mean()),
}
print(summary)
