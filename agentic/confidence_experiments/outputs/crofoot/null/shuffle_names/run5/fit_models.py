import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('crofoot.csv')

outcome = 'm_focal'  # binary
size_cols = ['f_other', 'win']
dist_cols = ['m_other', 'n_focal']

results = []

for focal_size in size_cols:
    other_size = [c for c in size_cols if c != focal_size][0]
    for dist_focal in dist_cols:
        dist_other = [c for c in dist_cols if c != dist_focal][0]

        df = _df.copy()
        df['relative_size'] = df[focal_size] - df[other_size]
        df['relative_size_log'] = np.log(df[focal_size] / df[other_size])
        df['relative_location'] = df[dist_focal] - df[dist_other]

        for name, cols in [('diff', ['relative_size','relative_location']), ('log', ['relative_size_log','relative_location'])]:
            X = sm.add_constant(df[cols])
            model = sm.Logit(df[outcome], X)
            try:
                res = model.fit(disp=False)
            except Exception as e:
                print('fit failed', focal_size, dist_focal, name, e)
                continue
            results.append({
                'focal_size': focal_size,
                'dist_focal': dist_focal,
                'spec': name,
                'aic': res.aic,
                'params': res.params.to_dict(),
                'pvalues': res.pvalues.to_dict(),
            })

# Sort by AIC
results_sorted = sorted(results, key=lambda x: x['aic'])

for r in results_sorted:
    print('\nModel', r['spec'], 'focal_size', r['focal_size'], 'dist_focal', r['dist_focal'], 'AIC', round(r['aic'],2))
    print(' params', r['params'])
    print(' pvalues', r['pvalues'])
