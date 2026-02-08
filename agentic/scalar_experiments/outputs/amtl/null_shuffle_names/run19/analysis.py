import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('amtl.csv')

# Remap columns based on observed values
# Mapping inferred from value ranges and category meanings
# sockets -> tooth class
# prob_male -> specimen id
# genus -> num_amtl (count missing teeth)
# age -> sockets (count observable)
# pop -> age (estimated age at death)
# num_amtl -> stdev_age (age uncertainty)
# stdev_age -> prob_male (sex probability)
# tooth_class -> genus (taxon)
# specimen -> population/region

df = pd.DataFrame({
    'tooth_class': raw['sockets'],
    'specimen_id': raw['prob_male'],
    'num_amtl': raw['genus'],
    'sockets': raw['age'],
    'age': raw['pop'],
    'stdev_age': raw['num_amtl'],
    'prob_male': raw['stdev_age'],
    'genus': raw['tooth_class'],
    'population': raw['specimen'],
})

# Keep valid rows where counts are sensible
valid = df[(df['sockets'] > 0) & (df['num_amtl'] >= 0)]
valid = valid[valid['num_amtl'] <= valid['sockets']].copy()

# Build binomial response
valid['failures'] = valid['sockets'] - valid['num_amtl']

# Fit binomial GLM with categorical genus and tooth class
# Use Homo sapiens as reference by ordering categories
valid['genus'] = pd.Categorical(valid['genus'],
                               categories=['Homo sapiens', 'Pan', 'Papio', 'Pongo'],
                               ordered=False)
valid['tooth_class'] = pd.Categorical(valid['tooth_class'],
                                     categories=['Anterior', 'Premolar', 'Posterior'],
                                     ordered=False)

# Drop any rows with missing categories
valid = valid.dropna(subset=['genus', 'tooth_class', 'age', 'prob_male'])

formula = 'num_amtl + failures ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=valid, family=sm.families.Binomial())
res = model.fit()

# Compute average predicted AMTL rate for Homo sapiens vs non-human primates
# Keep covariates as observed, set genus to target

def avg_pred_rate(target_genus):
    tmp = valid.copy()
    tmp['genus'] = target_genus
    pred = res.predict(tmp)
    return float(np.average(pred, weights=tmp['sockets']))

homo_rate = avg_pred_rate('Homo sapiens')
nonhuman_rates = [avg_pred_rate(g) for g in ['Pan', 'Papio', 'Pongo']]
nonhuman_rate = float(np.mean(nonhuman_rates))

diff = homo_rate - nonhuman_rate
odds_ratio = np.exp(res.params.get('C(genus)[T.Pan]', 0))

# Print key results
print('n_rows', len(valid))
print('homo_rate', homo_rate)
print('nonhuman_rate', nonhuman_rate)
print('diff', diff)
print('params', res.params)
print('pvalues', res.pvalues)

