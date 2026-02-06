import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('boxes.csv')

# Basic cleaning
# Keep relevant columns; drop rows with missing age or culture
_df = _df.copy()

# Ensure outcome coding as described (1,2,3)
# social reliance: chose demonstrated option (majority or minority)
_df['social'] = _df['y'].isin([2, 3]).astype(int)

# majority preference among demonstrated choices
_df['majority'] = (_df['y'] == 2).astype(int)

# Convert culture to categorical for modeling
_df['culture'] = _df['culture'].astype('category')

# Social reliance model
social_df = _df.dropna(subset=['age', 'culture', 'social'])

# Reduced model: age + culture
social_reduced = smf.logit('social ~ age + C(culture)', data=social_df).fit(disp=False)

# Full model with age*culture interaction
social_full = smf.logit('social ~ age * C(culture)', data=social_df).fit(disp=False)

lr_stat_social = 2 * (social_full.llf - social_reduced.llf)
lf_df_social = int(social_full.df_model - social_reduced.df_model)
if lf_df_social <= 0:
    p_lr_social = np.nan
else:
    p_lr_social = stats.chi2.sf(lr_stat_social, lf_df_social)

# Majority preference model (only among demonstrated choices)
maj_df = _df[_df['y'].isin([2, 3])].dropna(subset=['age', 'culture', 'majority'])

maj_reduced = smf.logit('majority ~ age + C(culture)', data=maj_df).fit(disp=False)
maj_full = smf.logit('majority ~ age * C(culture)', data=maj_df).fit(disp=False)

lr_stat_maj = 2 * (maj_full.llf - maj_reduced.llf)
lf_df_maj = int(maj_full.df_model - maj_reduced.df_model)
if lf_df_maj <= 0:
    p_lr_maj = np.nan
else:
    p_lr_maj = stats.chi2.sf(lr_stat_maj, lf_df_maj)

# Descriptive stats by culture and age bins
age_bins = [0, 6, 9, 12, 14, 100]
age_labels = ['4-6', '7-9', '10-12', '13-14', '15+']
_df['age_bin'] = pd.cut(_df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

social_rates = _df.groupby('culture', observed=True)['social'].mean()
maj_rates = _df[_df['y'].isin([2, 3])].groupby('culture', observed=True)['majority'].mean()

social_by_age = _df.groupby('age_bin', observed=True)['social'].mean()
maj_by_age = _df[_df['y'].isin([2, 3])].groupby('age_bin', observed=True)['majority'].mean()

print('N total:', len(_df))
print('N social model:', len(social_df))
print('N majority model:', len(maj_df))
print('\nSocial reliance reduced model (age + culture):')
print(social_reduced.summary2().tables[1].loc[['age']])
print('Culture terms p-values (min/max):', social_reduced.pvalues.filter(like='C(culture)').min(), social_reduced.pvalues.filter(like='C(culture)').max())
print('Interaction LR test p-value (age*culture):', p_lr_social)

print('\nMajority preference reduced model (age + culture):')
print(maj_reduced.summary2().tables[1].loc[['age']])
print('Culture terms p-values (min/max):', maj_reduced.pvalues.filter(like='C(culture)').min(), maj_reduced.pvalues.filter(like='C(culture)').max())
print('Interaction LR test p-value (age*culture):', p_lr_maj)

print('\nDescriptive variation (culture):')
print('Social reliance rate min/max:', social_rates.min(), social_rates.max())
print('Majority preference rate min/max:', maj_rates.min(), maj_rates.max())

print('\nDescriptive variation (age bins):')
print('Social reliance by age bin:')
print(social_by_age)
print('Majority preference by age bin:')
print(maj_by_age)

# Save results for potential downstream use
results = {
    'social_age_p': float(social_reduced.pvalues.loc['age']),
    'social_lr_p': float(p_lr_social) if not np.isnan(p_lr_social) else None,
    'maj_age_p': float(maj_reduced.pvalues.loc['age']),
    'maj_lr_p': float(p_lr_maj) if not np.isnan(p_lr_maj) else None,
    'social_rate_min': float(social_rates.min()),
    'social_rate_max': float(social_rates.max()),
    'maj_rate_min': float(maj_rates.min()),
    'maj_rate_max': float(maj_rates.max()),
}

pd.Series(results).to_json('analysis_results.json')
