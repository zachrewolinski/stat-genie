import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
df = pd.read_csv('amtl.csv')

# Create binary indicator for human
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS with genus categories (Homo sapiens as reference)
formula_genus = 'num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class) + sockets'
model_genus = smf.ols(formula_genus, data=df).fit(cov_type='HC3')

# OLS with human vs non-human
formula_human = 'num_amtl ~ is_human + age + prob_male + C(tooth_class) + sockets'
model_human = smf.ols(formula_human, data=df).fit(cov_type='HC3')

# Extract key results
human_coef = model_human.params['is_human']
human_pval = model_human.pvalues['is_human']

# Predicted means for genus at average covariates
avg_vals = {
    'age': df['age'].mean(),
    'prob_male': df['prob_male'].mean(),
    'sockets': df['sockets'].mean(),
}

preds = {}
for genus in df['genus'].unique():
    tmp = pd.DataFrame({
        'genus': [genus],
        'age': [avg_vals['age']],
        'prob_male': [avg_vals['prob_male']],
        'tooth_class': ['Anterior'],  # placeholder, will average over classes next
        'sockets': [avg_vals['sockets']],
    })
    # We'll average predictions over tooth classes to avoid picking one class
    pred_vals = []
    for tc in df['tooth_class'].unique():
        tmp['tooth_class'] = tc
        pred_vals.append(float(model_genus.predict(tmp)))
    preds[genus] = float(np.mean(pred_vals))

# Save outputs for later use
results = {
    'model_genus_summary': model_genus.summary().as_text(),
    'model_human_summary': model_human.summary().as_text(),
    'human_coef': float(human_coef),
    'human_pval': float(human_pval),
    'preds': preds,
    'n': int(df.shape[0])
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Human coef:', human_coef, 'p=', human_pval)
print('Preds:', preds)
