import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Negative binomial GLM for deaths

m_nb = smf.glm('alldeaths ~ masfem + category + wind + min', data=df,
              family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
print(m_nb.summary())

# Also with masfem_mturk
m_nb2 = smf.glm('alldeaths ~ masfem_mturk + category + wind + min', data=df,
               family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
print(m_nb2.summary())

# Binary gender
m_nb3 = smf.glm('alldeaths ~ gender_mf + category + wind + min', data=df,
               family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
print(m_nb3.summary())

# Save key p-values
import json

out = {
    'masfem_coef': m_nb.params.to_dict(),
    'masfem_p': m_nb.pvalues.to_dict(),
    'masfem_mturk_p': m_nb2.pvalues.to_dict(),
    'gender_mf_p': m_nb3.pvalues.to_dict(),
}

with open('analysis_nb_results.json', 'w') as f:
    json.dump(out, f, indent=2)

