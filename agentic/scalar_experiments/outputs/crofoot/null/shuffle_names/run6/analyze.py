import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Identify outcome (binary)
# From metadata, m_focal appears to be win indicator (0/1)
print('Outcome value counts (m_focal):')
print(df['m_focal'].value_counts())

# Candidate size columns: f_other (focal size), win (other size)
# Candidate distance columns: m_other (focal distance), n_focal (other distance)

# Create derived variables
focal_size = df['f_other']
other_size = df['win']
rel_size = focal_size - other_size

focal_dist = df['m_other']
other_dist = df['n_focal']
rel_dist = other_dist - focal_dist  # positive => focal closer to its center than other

analysis_df = df.copy()
analysis_df['rel_size'] = rel_size
analysis_df['rel_dist'] = rel_dist

# Basic correlations
print('\nDescriptives:')
print(analysis_df[['rel_size', 'rel_dist']].describe())

# Logistic regression
analysis_df = analysis_df.dropna(subset=['m_focal','rel_size','rel_dist'])

# Fit logistic regression
model = smf.logit('m_focal ~ rel_size + rel_dist', data=analysis_df).fit(disp=False)
print('\nLogit summary:')
print(model.summary())

# Also test each predictor separately for clarity
model_size = smf.logit('m_focal ~ rel_size', data=analysis_df).fit(disp=False)
model_dist = smf.logit('m_focal ~ rel_dist', data=analysis_df).fit(disp=False)

print('\nLogit size-only summary:')
print(model_size.summary())
print('\nLogit dist-only summary:')
print(model_dist.summary())

# Compute predicted probabilities effect sizes at +/-1 sd
for name, mod in [('both', model), ('size', model_size), ('dist', model_dist)]:
    params = mod.params
    if name == 'both':
        # baseline at mean
        mean_size = analysis_df['rel_size'].mean()
        mean_dist = analysis_df['rel_dist'].mean()
        # one sd shifts
        sd_size = analysis_df['rel_size'].std()
        sd_dist = analysis_df['rel_dist'].std()
        def predict(s, d):
            xb = params['Intercept'] + params['rel_size']*s + params['rel_dist']*d
            return 1/(1+np.exp(-xb))
        p_mean = predict(mean_size, mean_dist)
        p_size_up = predict(mean_size+sd_size, mean_dist)
        p_size_down = predict(mean_size-sd_size, mean_dist)
        p_dist_up = predict(mean_size, mean_dist+sd_dist)
        p_dist_down = predict(mean_size, mean_dist-sd_dist)
        print(f"\nPredicted probs (both model): mean={p_mean:.3f}, size+1sd={p_size_up:.3f}, size-1sd={p_size_down:.3f}, dist+1sd={p_dist_up:.3f}, dist-1sd={p_dist_down:.3f}")
    elif name == 'size':
        mean_size = analysis_df['rel_size'].mean()
        sd_size = analysis_df['rel_size'].std()
        def predict(s):
            xb = params['Intercept'] + params['rel_size']*s
            return 1/(1+np.exp(-xb))
        p_mean = predict(mean_size)
        p_up = predict(mean_size+sd_size)
        p_down = predict(mean_size-sd_size)
        print(f"\nPredicted probs (size model): mean={p_mean:.3f}, +1sd={p_up:.3f}, -1sd={p_down:.3f}")
    else:
        mean_dist = analysis_df['rel_dist'].mean()
        sd_dist = analysis_df['rel_dist'].std()
        def predict(d):
            xb = params['Intercept'] + params['rel_dist']*d
            return 1/(1+np.exp(-xb))
        p_mean = predict(mean_dist)
        p_up = predict(mean_dist+sd_dist)
        p_down = predict(mean_dist-sd_dist)
        print(f"\nPredicted probs (dist model): mean={p_mean:.3f}, +1sd={p_up:.3f}, -1sd={p_down:.3f}")
