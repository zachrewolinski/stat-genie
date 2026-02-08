import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: keep rows with valid counts
_df = _df.copy()
_df = _df[(~_df['num_amtl'].isna()) & (~_df['sockets'].isna())]
_df = _df[_df['sockets'] > 0]
_df = _df[_df['num_amtl'] >= 0]
_df = _df[_df['num_amtl'] <= _df['sockets']]

# Binary indicator for human
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Ensure categorical tooth_class
_df['tooth_class'] = _df['tooth_class'].astype('category')

# Build GLM binomial with successes/failures
_df['failures'] = _df['sockets'] - _df['num_amtl']

# Use formula to include categorical tooth_class
formula = 'num_amtl + failures ~ is_human + age + prob_male + C(tooth_class)'

model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial()
).fit()

# Extract coefficient for is_human
coef = model.params['is_human']
se = model.bse['is_human']
p_value = model.pvalues['is_human']

# Convert to odds ratio
odds_ratio = float(np.exp(coef))

# Predicted probability at average covariates for human vs non-human
mean_age = _df['age'].mean()
mean_male = _df['prob_male'].mean()
# For tooth_class, use observed proportions to average predictions
classes = _df['tooth_class'].cat.categories
class_probs = _df['tooth_class'].value_counts(normalize=True).reindex(classes).fillna(0)

def pred_prob(is_human):
    rows = []
    for cls in classes:
        rows.append({'is_human': is_human, 'age': mean_age, 'prob_male': mean_male, 'tooth_class': cls})
    pred = model.predict(pd.DataFrame(rows))
    return float((pred * class_probs.values).sum())

pred_nonhuman = pred_prob(0)
pred_human = pred_prob(1)

# Save key results to a json-like text for easy reading
with open('analysis_results.txt', 'w') as f:
    f.write(f"n_rows={len(_df)}\n")
    f.write(f"coef_is_human={coef}\n")
    f.write(f"se_is_human={se}\n")
    f.write(f"p_value_is_human={p_value}\n")
    f.write(f"odds_ratio_is_human={odds_ratio}\n")
    f.write(f"pred_prob_nonhuman={pred_nonhuman}\n")
    f.write(f"pred_prob_human={pred_human}\n")
