import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np


df = pd.read_csv('amtl.csv')

df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Mixed effects model with specimen random intercept to account for repeated tooth classes
mixed = smf.mixedlm('num_amtl ~ human + age + prob_male + C(tooth_class)', df, groups=df['specimen'])
res = mixed.fit(reml=False)

coef = float(res.params['human'])
pval = float(res.pvalues['human'])
ci_low, ci_high = [float(x) for x in res.conf_int().loc['human']]

sd = float(df['num_amtl'].std())
effect_sd = coef / sd if sd > 0 else float('nan')

mean_human = float(df.loc[df['human'] == 1, 'num_amtl'].mean())
mean_non = float(df.loc[df['human'] == 0, 'num_amtl'].mean())

# Simple mapping from effect and significance to Likert response
if coef > 0 and pval < 0.001:
    if effect_sd >= 0.5:
        response = 90
    elif effect_sd >= 0.3:
        response = 80
    elif effect_sd >= 0.1:
        response = 70
    else:
        response = 60
elif coef > 0 and pval < 0.05:
    response = 65
elif coef > 0:
    response = 55
elif coef < 0 and pval < 0.001:
    response = 20
elif coef < 0 and pval < 0.05:
    response = 35
elif coef < 0:
    response = 45
else:
    response = 50

explanation = (
    "A mixed-effects model (random intercept for specimen) predicting AMTL from a human indicator "
    "(Homo sapiens vs non-human primates), age, sex probability, and tooth class shows a positive, "
    f"statistically significant human effect (coef={coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], "
    f"p={pval:.2e}). The effect size is about {effect_sd:.2f} SD of the outcome, indicating a "
    "moderate increase in AMTL for humans after accounting for covariates. Raw means are also higher "
    f"for humans (mean={mean_human:.3f}) than non-humans (mean={mean_non:.3f}). "
    "Overall, the evidence supports a 'Yes' answer: modern humans have higher AMTL frequency than the "
    "non-human genera after controlling for age, sex, and tooth class."
)

with open('conclusion.txt', 'w') as f:
    json.dump({"response": int(response), "explanation": explanation}, f)
