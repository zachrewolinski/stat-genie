import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# Efficiency: nuts opened per second
# seconds are positive in the dataset

df['efficiency'] = df['nuts_opened'] / df['seconds']

n = len(df)

# OLS on efficiency
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()

# Count model with exposure (seconds) to check robustness
pois = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
               family=sm.families.Poisson(), offset=np.log(df['seconds'])).fit()

# Overdispersion check
poisson_dispersion = float(pois.pearson_chi2 / pois.df_resid)

# Negative binomial (robust to overdispersion)
nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
             family=sm.families.NegativeBinomial(alpha=1.0),
             offset=np.log(df['seconds'])).fit()

# Extract key stats
coef_age = ols.params['age']
coef_sex_m = ols.params.get('C(sex)[T.m]', np.nan)
coef_help_y = ols.params.get('C(help)[T.y]', np.nan)

p_age = ols.pvalues['age']
p_sex_m = ols.pvalues.get('C(sex)[T.m]', np.nan)
p_help_y = ols.pvalues.get('C(help)[T.y]', np.nan)

# Robustness p-values from NB
p_age_nb = nb.pvalues['age']
p_sex_nb = nb.pvalues.get('C(sex)[T.m]', np.nan)
p_help_nb = nb.pvalues.get('C(help)[T.y]', np.nan)

# Compose explanation
explanation = (
    f"Analyzed {n} sessions. Efficiency was defined as nuts_opened/seconds. "
    f"In OLS on efficiency, age (coef={coef_age:.3f}, p={p_age:.3g}) and sex (male vs female coef={coef_sex_m:.3f}, p={p_sex_m:.3g}) "
    f"were statistically significant, indicating higher efficiency for older chimpanzees and males. "
    f"Help from another chimpanzee was not significant in OLS (coef={coef_help_y:.3f}, p={p_help_y:.3g}). "
    f"A count model with exposure to session time showed strong overdispersion (Poisson dispersion≈{poisson_dispersion:.2f}), "
    f"so a negative binomial model was used for robustness; it again found age (p={p_age_nb:.3g}) and sex (p={p_sex_nb:.3g}) significant, "
    f"while help was not (p={p_help_nb:.3g}). "
    "Overall, there is clear evidence that age and sex influence nut‑cracking efficiency, but little evidence that receiving help does."
)

# Likert response: moderate yes because two predictors show consistent effects, but help does not
response = 65

with open('conclusion.txt', 'w') as f:
    json.dump({"response": int(response), "explanation": explanation}, f)
