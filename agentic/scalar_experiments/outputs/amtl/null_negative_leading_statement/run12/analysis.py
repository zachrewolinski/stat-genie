import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
# Ensure sockets >0
_df = _df[_df['sockets'] > 0].copy()

# Create human indicator
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Create design matrix with covariates
# Age, prob_male, tooth_class (categorical), genus (as human indicator)
X = _df[['age', 'prob_male', 'is_human', 'tooth_class']].copy()
X = pd.get_dummies(X, columns=['tooth_class'], drop_first=True)
X = sm.add_constant(X, has_constant='add')

# Response as proportion with binomial weights
endog = _df['num_amtl']
weights = _df['sockets']

# Fit GLM binomial
model = sm.GLM(endog, X, family=sm.families.Binomial(), freq_weights=weights)
res = model.fit()

# Extract human coefficient
coef = res.params['is_human']
se = res.bse['is_human']
z = coef / se
pval = res.pvalues['is_human']

# Convert to odds ratio
odds_ratio = float(np.exp(coef))

# Compute predicted rates for human vs non-human at mean covariates
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()

# For tooth_class, we can set to overall distribution or pick reference (Anterior baseline)
# Here we compute average predicted probability weighting by observed tooth_class frequencies
classes = _df['tooth_class'].unique()

def predict_prob(is_human):
    probs = []
    weights_tc = []
    for tc in classes:
        row = {
            'const': 1.0,
            'age': mean_age,
            'prob_male': mean_prob_male,
            'is_human': is_human,
            # dummies (drop_first so baseline is first sorted)
        }
        for col in X.columns:
            if col.startswith('tooth_class_'):
                row[col] = 1.0 if col == f'tooth_class_{tc}' else 0.0
        # If tc is baseline, all dummies zero
        row = pd.Series(row)
        # Align
        row = row.reindex(X.columns, fill_value=0.0)
        p = float(res.predict(row))
        probs.append(p)
        weights_tc.append((_df['tooth_class'] == tc).mean())
    return float(np.average(probs, weights=weights_tc))

pred_human = predict_prob(1)
pred_nonhuman = predict_prob(0)

# Save key outputs
summary = {
    'coef_is_human': coef,
    'se_is_human': se,
    'z_is_human': z,
    'p_is_human': pval,
    'odds_ratio_is_human': odds_ratio,
    'pred_prob_human': pred_human,
    'pred_prob_nonhuman': pred_nonhuman,
}

for k, v in summary.items():
    print(f"{k}: {v}")

