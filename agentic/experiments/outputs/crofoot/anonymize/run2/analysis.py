import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("crofoot.csv")

# Outcome: focal group win (1) vs other win (0)
y = _df["feature4"].astype(int)

# Relative group size: difference in total group size
size_diff = _df["feature7"] - _df["feature8"]

# Contest location: who is closer to its home range center
# Positive means focal is closer than the other group
location_adv = _df["feature6"] - _df["feature5"]

X = pd.DataFrame({
    "size_diff": size_diff,
    "location_adv": location_adv,
})
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X).fit(disp=False)

# Also compute a reduced model with each predictor alone for robustness
model_size = sm.Logit(y, sm.add_constant(pd.DataFrame({"size_diff": size_diff}))).fit(disp=False)
model_loc = sm.Logit(y, sm.add_constant(pd.DataFrame({"location_adv": location_adv}))).fit(disp=False)

print("Full model coefficients and p-values:")
print(pd.DataFrame({
    "coef": model.params,
    "p_value": model.pvalues
}))

print("\nSize-only model p-value:", model_size.pvalues["size_diff"])
print("Location-only model p-value:", model_loc.pvalues["location_adv"])

# Save key results to a small CSV for inspection if needed
results = pd.DataFrame({
    "term": model.params.index,
    "coef": model.params.values,
    "p_value": model.pvalues.values,
})
results.to_csv("analysis_results.csv", index=False)
