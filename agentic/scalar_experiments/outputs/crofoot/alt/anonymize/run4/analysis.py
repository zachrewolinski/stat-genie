import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Define variables
# Outcome: focal group win (1) vs loss (0)
df['win'] = df['feature4']

# Relative group size: focal size - other size
df['rel_size'] = df['feature7'] - df['feature8']

# Contest location: relative distance to home-range centers
# Negative values mean contest is closer to focal group's center (focal distance smaller)
df['rel_location'] = df['feature5'] - df['feature6']

# Prepare design matrix
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)
y = df['win']

# Fit logistic regression (GLM binomial for stability)
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Extract key stats
params = result.params
pvalues = result.pvalues

# Odds ratios for interpretability
odds_ratios = np.exp(params)

# Summaries for reasoning
n = len(df)

summary = {
    'n': n,
    'params': params.to_dict(),
    'pvalues': pvalues.to_dict(),
    'odds_ratios': odds_ratios.to_dict()
}

# Determine evidence strength for each predictor
size_p = pvalues['rel_size']
loc_p = pvalues['rel_location']

# Simple heuristic for Likert response
# Start from neutral (50) and adjust based on significance and effect direction
response = 50

# Location effect
if loc_p < 0.001:
    response += 30
elif loc_p < 0.01:
    response += 20
elif loc_p < 0.05:
    response += 10

# Size effect
if size_p < 0.001:
    response += 25
elif size_p < 0.01:
    response += 15
elif size_p < 0.05:
    response += 8

# If both non-significant, reduce
if loc_p >= 0.05 and size_p >= 0.05:
    response -= 20

# Cap between 0 and 100
response = int(max(0, min(100, round(response))))

# Build explanation
explanation = (
    f"Analyzed {n} contests with a logistic regression predicting focal win (feature4) "
    f"from relative group size (feature7 - feature8) and relative contest location "
    f"(feature5 - feature6). The model estimates for rel_size: coef={params['rel_size']:.3f}, "
    f"OR={odds_ratios['rel_size']:.3f}, p={size_p:.4f}; "
    f"for rel_location: coef={params['rel_location']:.4f}, OR={odds_ratios['rel_location']:.4f}, "
    f"p={loc_p:.4f}. A negative rel_location coefficient means contests closer to the focal group's "
    f"home-range center (smaller focal distance) increase win probability. "
    f"The Likert response reflects statistical significance and effect sizes of both predictors."
)

# Write conclusion
with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'explanation': explanation}, f)

# Also print summary for debugging
print(json.dumps(summary, indent=2))
