import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import logit

# Load data
path = 'boxes.csv'
df = pd.read_csv(path)

# Map columns for readability
# feature1: outcome (1=unchosen,2=majority,3=minority)
# feature3: age
# feature5: site

# Create indicators

df = df.copy()
df['chosen_demo'] = df['feature1'].isin([2,3]).astype(int)  # reliance on social info
# majority preference among those who chose a demonstrated option
sub = df[df['chosen_demo'] == 1].copy()
sub['majority_choice'] = (sub['feature1'] == 2).astype(int)

# Prepare categorical site
# Use C(feature5) in formula for categorical

# Model 1: reliance on social info ~ age + site
model1 = logit('chosen_demo ~ feature3 + C(feature5)', data=df).fit(disp=False)

# Model 2: majority preference ~ age + site (only among demonstrated choices)
model2 = logit('majority_choice ~ feature3 + C(feature5)', data=sub).fit(disp=False)

# Function to get joint significance of site via likelihood ratio test

def lr_test(model_full, model_reduced):
    llf_full = model_full.llf
    llf_reduced = model_reduced.llf
    df_diff = model_full.df_model - model_reduced.df_model
    lr_stat = 2 * (llf_full - llf_reduced)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

# Reduced models without site and without age
model1_no_site = logit('chosen_demo ~ feature3', data=df).fit(disp=False)
model1_no_age = logit('chosen_demo ~ C(feature5)', data=df).fit(disp=False)

model2_no_site = logit('majority_choice ~ feature3', data=sub).fit(disp=False)
model2_no_age = logit('majority_choice ~ C(feature5)', data=sub).fit(disp=False)

# LR tests
m1_site_lr = lr_test(model1, model1_no_site)
m1_age_lr = lr_test(model1, model1_no_age)

m2_site_lr = lr_test(model2, model2_no_site)
m2_age_lr = lr_test(model2, model2_no_age)

# Collect p-values
pvals = {
    'reliance_site_p': m1_site_lr[2],
    'reliance_age_p': m1_age_lr[2],
    'majority_site_p': m2_site_lr[2],
    'majority_age_p': m2_age_lr[2],
}

# Create evidence score: average of four components (site/age for both outcomes)
# Each component contributes 25 points if p<0.05, 15 points if p<0.10, 0 otherwise.
# Total between 0 and 100. Convert to -70..70 with linear mapping (0->-70, 100->70).
score = 0
for key, p in pvals.items():
    if p < 0.05:
        score += 25
    elif p < 0.10:
        score += 15

scalar = int(round((score/100)*140 - 70))

# Save scalar
with open('conclusion.txt','w') as f:
    f.write(str(scalar))

# Print summary for inspection
print('p-values:', pvals)
print('score:', score, 'scalar:', scalar)
