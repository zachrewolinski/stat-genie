import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Create human indicator
human_label = 'Homo sapiens'
df['human'] = (df['genus'] == human_label).astype(int)

# Basic counts
summary = df.groupby('genus')['num_amtl'].agg(['mean','std','count'])

# OLS with cluster-robust SEs by specimen to account for repeated measures
model = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=df)
fit = model.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

coef = fit.params['human']
se = fit.bse['human']
pval = fit.pvalues['human']
ci_low, ci_high = fit.conf_int().loc['human']

# Genus categorical model with explicit non-human reference (Pan) for context
model_genus = smf.ols('num_amtl ~ C(genus, Treatment(reference="Pan")) + age + prob_male + C(tooth_class)', data=df)
fit_genus = model_genus.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

coef_hs = fit_genus.params.get('C(genus, Treatment(reference="Pan"))[T.Homo sapiens]')
se_hs = fit_genus.bse.get('C(genus, Treatment(reference="Pan"))[T.Homo sapiens]')
pval_hs = fit_genus.pvalues.get('C(genus, Treatment(reference="Pan"))[T.Homo sapiens]')
ci_hs = fit_genus.conf_int().loc['C(genus, Treatment(reference="Pan"))[T.Homo sapiens]']

# Print results
print('Mean num_amtl by genus:')
print(summary)
print('\nHuman vs non-human OLS (clustered by specimen):')
print(f'coef={coef:.4f}, se={se:.4f}, p={pval:.4g}, 95% CI=({ci_low:.4f}, {ci_high:.4f})')

print('\nGenus categorical model (Homo vs Pan baseline):')
print(f'coef={coef_hs:.4f}, se={se_hs:.4f}, p={pval_hs:.4g}, 95% CI=({ci_hs[0]:.4f}, {ci_hs[1]:.4f})')

# Save key results
import json
with open('analysis_results.json','w') as f:
    json.dump({
        'summary': summary.to_dict(),
        'human_coef': coef,
        'human_se': se,
        'human_p': pval,
        'human_ci': [ci_low, ci_high],
        'hs_coef_vs_pan': coef_hs,
        'hs_se_vs_pan': se_hs,
        'hs_p_vs_pan': pval_hs,
        'hs_ci_vs_pan': [ci_hs[0], ci_hs[1]],
        'n': len(df)
    }, f, indent=2)
