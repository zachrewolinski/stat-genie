import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load
_df = pd.read_csv('amtl.csv')

# Map columns
_df = _df.rename(columns={
    'sockets': 'tooth_class',
    'prob_male': 'specimen_id',
    'genus': 'num_missing',
    'age': 'num_sockets',
    'pop': 'age_est',
    'num_amtl': 'age_sd',
    'stdev_age': 'prob_male',
    'tooth_class': 'genus',
    'specimen': 'population'
})

# Basic checks
_df = _df.dropna(subset=['num_missing','num_sockets','age_est','prob_male','tooth_class','genus'])
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Ensure valid counts
_df = _df[(_df['num_missing'] >= 0) & (_df['num_sockets'] > 0) & (_df['num_missing'] <= _df['num_sockets'])]

# GLM binomial with counts
_df['num_present'] = _df['num_sockets'] - _df['num_missing']

formula = 'num_missing + num_present ~ is_human + age_est + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial()).fit()

print(model.summary())

# Compute odds ratio and p-value for is_human
coef = model.params['is_human']
se = model.bse['is_human']
p = model.pvalues['is_human']

odds_ratio = float(np.exp(coef))

print('is_human coef', coef)
print('odds_ratio', odds_ratio)
print('p', p)

# Compute predicted probabilities for human vs nonhuman at mean covariates
mean_age = _df['age_est'].mean()
mean_prob_male = _df['prob_male'].mean()

# For each tooth class, compute average; then average predictions
classes = _df['tooth_class'].unique()

def predict_prob(is_human):
    preds = []
    for tc in classes:
        row = {
            'is_human': is_human,
            'age_est': mean_age,
            'prob_male': mean_prob_male,
            'tooth_class': tc,
        }
        # Use model to predict using design matrix
        pred = model.predict(pd.DataFrame([row]))[0]
        preds.append(pred)
    return float(np.mean(preds))

ph = predict_prob(1)
pnh = predict_prob(0)
print('pred human', ph, 'pred nonhuman', pnh, 'diff', ph-pnh)

