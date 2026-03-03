import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
file_path = 'amtl.csv'
df = pd.read_csv(file_path)

# Map columns to meaning (based on patterns and info.json)
# response: genus column (appears to be missing teeth count per class, standardized)
# tooth class: sockets column
# genus/species: tooth_class column
# specimen id: prob_male column
# age at death: pop column
# sex/prob male: stdev_age column

# Create binary human indicator
_df = df.copy()
_df['is_human'] = (_df['tooth_class'] == 'Homo sapiens').astype(int)

# Treat tooth class and species as categorical
_df['tooth_class_cat'] = _df['sockets'].astype('category')
_df['species_cat'] = _df['tooth_class'].astype('category')

# Center/scale age for stability
_df['age_at_death'] = _df['pop']
_df['sex_prob_male'] = _df['stdev_age']

# Response
_df['y'] = _df['genus']

# Model: y ~ is_human + age + sex + tooth class
# Use clustered SE by specimen ID
model = smf.ols('y ~ is_human + age_at_death + sex_prob_male + tooth_class_cat', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['prob_male']}
)

print(model.summary())

# Also model with species categories (Homo, Pan, Papio, Pongo) for comparison
model_species = smf.ols('y ~ species_cat + age_at_death + sex_prob_male + tooth_class_cat', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['prob_male']}
)

print("\nSpecies model summary (coeffs):")
print(model_species.summary())

# Compute means by species
means = _df.groupby('tooth_class')['y'].mean().sort_values(ascending=False)
print("\nMean y by species:")
print(means)

# Compare Homo vs non-human via t-test on residualized y (control for covariates)
# Use model without is_human but with controls to get residuals
controls_model = smf.ols('y ~ age_at_death + sex_prob_male + tooth_class_cat', data=_df).fit()
_df['resid'] = controls_model.resid

human_resid = _df.loc[_df['is_human'] == 1, 'resid']
nonhuman_resid = _df.loc[_df['is_human'] == 0, 'resid']

from scipy import stats

t_stat, p_val = stats.ttest_ind(human_resid, nonhuman_resid, equal_var=False)
print(f"\nT-test on residuals (Homo vs non-human): t={t_stat:.3f}, p={p_val:.4f}")

# effect size (Cohen's d)
mean_diff = human_resid.mean() - nonhuman_resid.mean()
pooled_sd = np.sqrt(((human_resid.var(ddof=1) + nonhuman_resid.var(ddof=1)) / 2))
cohen_d = mean_diff / pooled_sd
print(f"Mean residual diff: {mean_diff:.3f}, Cohen's d: {cohen_d:.3f}")

# Save key outputs for reporting
with open('analysis_results.txt', 'w') as f:
    f.write(str(model.summary()))
    f.write("\n\nSpecies model\n")
    f.write(str(model_species.summary()))
    f.write("\n\nMeans by species\n")
    f.write(str(means))
    f.write(f"\n\nT-test residuals t={t_stat:.3f}, p={p_val:.4f}, d={cohen_d:.3f}\n")
