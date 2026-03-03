import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Compute missing proportion and clip
# feature3: missing teeth count for class (may include negatives in anonymized data)
# feature4: observable sockets

df['missing_prop'] = df['feature3'] / df['feature4']
df['missing_prop_clip'] = df['missing_prop'].clip(0, 1)

# Human indicator

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Descriptive stats

desc = df.groupby('feature8').agg(
    n=('missing_prop', 'size'),
    mean_prop=('missing_prop', 'mean'),
    mean_clip=('missing_prop_clip', 'mean'),
)
print('Descriptive by genus')
print(desc)

# GLM human vs non-human

glm = smf.glm(
    'missing_prop_clip ~ is_human + feature5 + feature7 + C(feature1)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['feature4'],
).fit()

print('\nGLM human vs non-human (binomial on clipped proportion, weights=feature4)')
print(glm.summary())

# Odds ratio for is_human
coef = glm.params['is_human']
se = glm.bse['is_human']
or_val = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)
print(f"Odds ratio is_human: {or_val:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f})")

# GLM genus model

glm_genus = smf.glm(
    'missing_prop_clip ~ C(feature8) + feature5 + feature7 + C(feature1)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['feature4'],
).fit()

print('\nGLM full genus')
print(glm_genus.summary())

# Predicted probabilities at mean covariates for each genus
mean_age = df['feature5'].mean()
mean_sex = df['feature7'].mean()
# Use most common tooth class for baseline, then average over tooth class distribution
# We'll compute predictions by weighting tooth class frequencies

class_freq = df['feature1'].value_counts(normalize=True)

def predict_for_genus(genus):
    probs = []
    weights = []
    for tooth_class, w in class_freq.items():
        row = pd.DataFrame({
            'feature8': [genus],
            'feature5': [mean_age],
            'feature7': [mean_sex],
            'feature1': [tooth_class],
        })
        pred = glm_genus.predict(row)[0]
        probs.append(pred)
        weights.append(w)
    return float(np.average(probs, weights=weights))

preds = {g: predict_for_genus(g) for g in df['feature8'].unique()}
print('\nPredicted AMTL probability at mean covariates (averaged over tooth class)')
for g, p in preds.items():
    print(f"{g}: {p:.4f}")
