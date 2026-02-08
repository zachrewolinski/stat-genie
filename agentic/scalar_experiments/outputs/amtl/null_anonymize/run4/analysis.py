import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Rename for clarity

df = df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'total',
    'feature5': 'age',
    'feature6': 'age_unc',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
})

# Basic cleaning: ensure valid totals

df = df[(df['total'] > 0) & (df['missing'] >= 0) & (df['missing'] <= df['total'])].copy()

# Binary indicator for humans

df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Failures for binomial counts

df['fail'] = df['total'] - df['missing']

# Fit binomial GLM for AMTL counts
model = smf.glm(
    formula='missing + fail ~ human + age + sex + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
).fit()

# Extract human effect
coef_human = model.params['human']
se_human = model.bse['human']
z_human = coef_human / se_human
p_human = model.pvalues['human']

# Odds ratio
or_human = float(np.exp(coef_human))

# Predicted probabilities at mean covariates for human vs non-human
mean_age = df['age'].mean()
mean_sex = df['sex'].mean()
# use most common tooth_class for reference
ref_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'human': [0, 1],
    'age': [mean_age, mean_age],
    'sex': [mean_sex, mean_sex],
    'tooth_class': [ref_tooth, ref_tooth]
})

pred = model.predict(pred_df)
prob_nonhuman, prob_human = pred.tolist()

# Map to Likert scalar [-100, 100]
# Use direction from coef; strength from |z| and effect size in probability difference
prob_diff = prob_human - prob_nonhuman

# z-based confidence component
conf = np.tanh(abs(z_human) / 3.0)  # ~0.76 at z=3

# effect size component based on probability difference (cap at 0.25)
size = min(abs(prob_diff) / 0.25, 1.0)

score = (conf * 0.6 + size * 0.4) * 100
score = float(np.sign(coef_human) * score)

# Round to integer within [-100, 100]
score_int = int(np.round(np.clip(score, -100, 100)))

print({
    'coef_human': coef_human,
    'se_human': se_human,
    'z_human': z_human,
    'p_human': p_human,
    'odds_ratio': or_human,
    'prob_nonhuman': prob_nonhuman,
    'prob_human': prob_human,
    'prob_diff': prob_diff,
    'score': score,
    'score_int': score_int,
})

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score_int))
