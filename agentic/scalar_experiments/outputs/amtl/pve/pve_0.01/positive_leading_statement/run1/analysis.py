import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic checks
summary = {
    'rows': len(_df),
    'num_amtl_min': float(_df['num_amtl'].min()),
    'num_amtl_max': float(_df['num_amtl'].max()),
    'num_amtl_mean': float(_df['num_amtl'].mean()),
    'num_amtl_std': float(_df['num_amtl'].std()),
}

# Regression: num_amtl ~ genus + age + prob_male + tooth_class
# Use Homo sapiens as reference? We'll set categorical ordering so Homo sapiens is base.
_df['genus'] = pd.Categorical(_df['genus'], categories=['Homo sapiens', 'Pan', 'Pongo', 'Papio'])
_df['tooth_class'] = pd.Categorical(_df['tooth_class'], categories=['Anterior', 'Posterior', 'Premolar'])

model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=_df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen']})

# Also check including sockets to see robustness
model_sockets = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class) + sockets', data=_df)
res_sockets = model_sockets.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen']})

# Joint test for genus effects (Pan, Pongo, Papio vs Homo sapiens)
genus_terms = [
    'C(genus)[T.Pan] = 0',
    'C(genus)[T.Pongo] = 0',
    'C(genus)[T.Papio] = 0',
]
genus_test = res.f_test(genus_terms)
genus_test_sockets = res_sockets.f_test(genus_terms)

# Extract key comparisons: coefficients for non-human vs Homo sapiens
coef = res.params
pvals = res.pvalues
coef_s = res_sockets.params
pvals_s = res_sockets.pvalues

out = {
    'summary': summary,
    'coef': coef.to_dict(),
    'pvals': pvals.to_dict(),
    'coef_sockets': coef_s.to_dict(),
    'pvals_sockets': pvals_s.to_dict(),
    'genus_f_pvalue': float(genus_test.pvalue),
    'genus_f_stat': float(genus_test.fvalue),
    'genus_f_pvalue_sockets': float(genus_test_sockets.pvalue),
    'genus_f_stat_sockets': float(genus_test_sockets.fvalue),
    'r2': float(res.rsquared),
    'r2_sockets': float(res_sockets.rsquared),
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)
