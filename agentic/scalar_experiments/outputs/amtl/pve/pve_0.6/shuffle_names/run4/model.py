import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('amtl.csv')

# Map columns to their semantic meaning per info.json while avoiding duplicate names
# sockets -> tooth_class
# prob_male -> specimen_id
# genus -> amtl (numeric response)
# age -> sockets_count
# pop -> age_at_death
# num_amtl -> age_uncertainty
# stdev_age -> prob_male
# tooth_class -> genus (Homo/Pan/Papio/Pongo)
# specimen -> population

df = raw.rename(columns={
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id',
    'genus': 'amtl',
    'age': 'sockets_count',
    'pop': 'age_at_death',
    'num_amtl': 'age_uncertainty',
    'stdev_age': 'prob_male',
    'tooth_class': 'genus',
    'specimen': 'population'
})

# Ensure categorical ordering

df['genus'] = pd.Categorical(
    df['genus'],
    categories=['Homo sapiens', 'Pan', 'Papio', 'Pongo']
)

df['tooth_class'] = pd.Categorical(
    df['tooth_class'],
    categories=['Anterior', 'Posterior', 'Premolar']
)

# Fit linear model controlling for age, sex, tooth class
model = smf.ols('amtl ~ C(genus) + age_at_death + prob_male + C(tooth_class)', data=df).fit()

# Contrast: Homo sapiens vs average of non-human genera
param_names = model.params.index.tolist()
contrast = np.zeros(len(param_names))

for g in ['Pan', 'Papio', 'Pongo']:
    term = f'C(genus)[T.{g}]'
    if term in param_names:
        contrast[param_names.index(term)] = -1/3

# Homo - avg(nonhuman) equals negative mean of nonhuman coefficients (baseline Homo)
res = model.t_test(contrast)

estimate = float(res.effect)  # difference Homo - avg(nonhuman)
p_value = float(res.pvalue)

# Also compute adjusted means (at mean covariates, averaged across tooth classes)
mean_age = df['age_at_death'].mean()
mean_prob_male = df['prob_male'].mean()

# Build prediction grid
pred_rows = []
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    for tooth_class in ['Anterior', 'Posterior', 'Premolar']:
        pred_rows.append({
            'genus': genus,
            'age_at_death': mean_age,
            'prob_male': mean_prob_male,
            'tooth_class': tooth_class
        })

pred_df = pd.DataFrame(pred_rows)
pred_df['pred'] = model.predict(pred_df)

mean_preds = pred_df.groupby('genus')['pred'].mean().to_dict()

print('Estimate (Homo - avg nonhuman):', estimate)
print('p_value:', p_value)
print('Adjusted mean predictions:', mean_preds)

# Save key outputs for later use
with open('model_results.txt', 'w') as f:
    f.write(f'estimate_homo_minus_nonhuman={estimate}\n')
    f.write(f'p_value={p_value}\n')
    for k, v in mean_preds.items():
        f.write(f'pred_{k}={v}\n')
