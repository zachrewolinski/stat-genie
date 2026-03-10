import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

# Load data

df = pd.read_csv('panda_nuts.csv')

# Efficiency as rate of nuts opened per second

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Encode categoricals

df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# Use negative binomial GLM for count data with offset for exposure time
# This models nuts_opened per unit time, i.e., efficiency, while addressing overdispersion.

df['log_seconds'] = np.log(df['seconds'])
model_nb = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=df['log_seconds']
).fit()

# Extract key stats
params = model_nb.params
pvalues = model_nb.pvalues

# Group summaries
counts = df[['sex','help']].value_counts().sort_index()
mean_eff_by_sex = df.groupby('sex')['efficiency'].mean()
mean_eff_by_help = df.groupby('help')['efficiency'].mean()

# Build explanation
explanation = (
    f"Analysis used a negative binomial GLM for nuts_opened with log(seconds) as an offset "
    f"to model nut-cracking efficiency (rate), n={len(df)}. "
    f"Sex shows a statistically significant association with efficiency: males have lower rates "
    f"(coef={params['C(sex)[T.m]']:.3f}, p={pvalues['C(sex)[T.m]']:.3f}). "
    f"Age is not significant (coef={params['age']:.3f}, p={pvalues['age']:.3f}). "
    f"Receiving help is not significant (coef={params['C(help)[T.y]']:.3f}, p={pvalues['C(help)[T.y]']:.3f}); "
    f"help was rare (counts by sex/help: {counts.to_dict()}). "
    f"Mean efficiency by sex (nuts/sec): {mean_eff_by_sex.to_dict()}; "
    f"by help: {mean_eff_by_help.to_dict()}. "
    f"Overall, evidence supports an influence of sex but not age or help on efficiency."
)

# Likert response: partial evidence (sex only) -> leaning 'No' for all three jointly
response = 40

out = {"response": int(response), "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(out, f)

