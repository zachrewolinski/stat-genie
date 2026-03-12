import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Identify columns
win_col = 'm_focal'  # only 0/1 column
size_cols = ['f_other', 'win']
dist_cols = ['m_other', 'n_focal']

# Try combinations to orient relative size/location with positive coefficients if possible
results = []

for size_focal in size_cols:
    size_other = [c for c in size_cols if c != size_focal][0]
    rel_size = df[size_focal] - df[size_other]

    for dist_focal in dist_cols:
        dist_other = [c for c in dist_cols if c != dist_focal][0]
        # relative location: positive means contest closer to focal (other distance larger)
        rel_loc = df[dist_other] - df[dist_focal]

        X = pd.DataFrame({
            'rel_size': rel_size,
            'rel_loc': rel_loc,
        })
        X = sm.add_constant(X)
        y = df[win_col]
        try:
            model = sm.Logit(y, X).fit(disp=False)
        except Exception as e:
            results.append((size_focal, dist_focal, None, str(e)))
            continue
        results.append((size_focal, dist_focal, model, None))

# Print summary for each model
for size_focal, dist_focal, model, err in results:
    print('\n=== focal_size:', size_focal, 'focal_dist:', dist_focal, '===')
    if err:
        print('Error:', err)
        continue
    print(model.summary())
    params = model.params
    pvals = model.pvalues
    print('coeffs:', params.to_dict())
    print('pvals:', pvals.to_dict())

# Choose model with positive coefficients for both predictors if possible
best = None
for size_focal, dist_focal, model, err in results:
    if err or model is None:
        continue
    params = model.params
    if params['rel_size'] > 0 and params['rel_loc'] > 0:
        # choose with smallest combined p-values
        score = params.index
        combined_p = model.pvalues['rel_size'] + model.pvalues['rel_loc']
        if best is None or combined_p < best[0]:
            best = (combined_p, size_focal, dist_focal, model)

if best:
    combined_p, size_focal, dist_focal, model = best
    print('\nBEST (positive coeffs) focal_size:', size_focal, 'focal_dist:', dist_focal)
    print(model.summary())
else:
    # fallback: pick smallest combined p
    best = None
    for size_focal, dist_focal, model, err in results:
        if err or model is None:
            continue
        combined_p = model.pvalues['rel_size'] + model.pvalues['rel_loc']
        if best is None or combined_p < best[0]:
            best = (combined_p, size_focal, dist_focal, model)
    if best:
        combined_p, size_focal, dist_focal, model = best
        print('\nBEST (min combined p) focal_size:', size_focal, 'focal_dist:', dist_focal)
        print(model.summary())

# Also compute simple correlations for rel_size and rel_loc with win for selected mapping
if best:
    _, size_focal, dist_focal, model = best
    size_other = [c for c in size_cols if c != size_focal][0]
    dist_other = [c for c in dist_cols if c != dist_focal][0]
    rel_size = df[size_focal] - df[size_other]
    rel_loc = df[dist_other] - df[dist_focal]
    print('\nRel_size mean:', rel_size.mean(), 'std:', rel_size.std())
    print('Rel_loc mean:', rel_loc.mean(), 'std:', rel_loc.std())
    print('Win rate:', df[win_col].mean())
    print('Correlation rel_size-win:', np.corrcoef(rel_size, df[win_col])[0,1])
    print('Correlation rel_loc-win:', np.corrcoef(rel_loc, df[win_col])[0,1])

