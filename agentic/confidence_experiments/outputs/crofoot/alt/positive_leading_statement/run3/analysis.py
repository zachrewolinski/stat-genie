import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('crofoot.csv')

# Create relative predictors
DF['size_diff'] = DF['n_focal'] - DF['n_other']
# Positive means focal group is closer to its home-range center than the other group
DF['location_adv'] = DF['dist_other'] - DF['dist_focal']

# Logistic regression: win ~ size_diff + location_adv
model = smf.logit('win ~ size_diff + location_adv', data=DF)
result = model.fit(disp=False)

# Compute marginal effects at means
marginal = result.get_margeff(at='mean')

# Summaries
summary_df = pd.DataFrame({
    'coef': result.params,
    'se': result.bse,
    'z': result.tvalues,
    'p': result.pvalues
})

# Effect sizes using one SD change in predictors
sd_size = DF['size_diff'].std()
sd_loc = DF['location_adv'].std()

# Predicted probability at mean and with +1 SD for each predictor (holding other at mean)
mean_vals = DF[['size_diff', 'location_adv']].mean()

def predict_prob(size_diff, location_adv):
    x = pd.DataFrame({'size_diff': [size_diff], 'location_adv': [location_adv]})
    return float(result.predict(x)[0])

base_prob = predict_prob(mean_vals['size_diff'], mean_vals['location_adv'])
prob_size_plus = predict_prob(mean_vals['size_diff'] + sd_size, mean_vals['location_adv'])
prob_loc_plus = predict_prob(mean_vals['size_diff'], mean_vals['location_adv'] + sd_loc)

output = {
    'n': len(DF),
    'model_params': summary_df.to_dict(orient='index'),
    'marginal_effects': marginal.summary_frame().to_dict(orient='index'),
    'sd_size_diff': sd_size,
    'sd_location_adv': sd_loc,
    'base_prob': base_prob,
    'prob_size_plus1sd': prob_size_plus,
    'prob_location_plus1sd': prob_loc_plus,
}

# Save for inspection
import json
with open('analysis_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print(result.summary())
print('\nMarginal effects at mean:\n', marginal.summary())
print('\nBase prob:', base_prob)
print('Prob +1SD size_diff:', prob_size_plus)
print('Prob +1SD location_adv:', prob_loc_plus)
