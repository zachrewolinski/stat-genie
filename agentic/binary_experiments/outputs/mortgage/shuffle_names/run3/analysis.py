import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "mortgage.csv"
df = pd.read_csv(csv_path)

# Column names are shuffled. Use metadata hints:
# - 'denied_PMI' column description indicates it is female (gender)
# - 'self_employed' column description indicates it is the denial outcome
# - 'deny' appears to be the inverse of denial (redundant)
# - 'bad_history' appears to be an ID-like column (0..2379)

# Define outcome and gender indicator
# denied = 1 if application denied, 0 if approved

df = df.copy()
df["denied"] = df["self_employed"]
df["female_ind"] = df["denied_PMI"]

# Build predictor set (all other columns except ID/redundant/outcome)
exclude = {"self_employed", "deny", "bad_history", "denied", "female_ind"}
predictors = [c for c in df.columns if c not in exclude]

# Drop rows with missing values in relevant columns
model_df = df[["denied", "female_ind"] + predictors].dropna()

# Denial rates by gender
rates = model_df.groupby("female_ind")["denied"].mean().rename({0.0: "male", 1.0: "female"})

# Unadjusted logit: denial ~ female
X_unadj = sm.add_constant(model_df[["female_ind"]])
model_unadj = sm.Logit(model_df["denied"], X_unadj).fit(disp=False)

# Adjusted logit with other covariates
other_preds = [p for p in predictors if p != "denied_PMI"]
X_adj = sm.add_constant(model_df[["female_ind"] + other_preds])
model_adj = sm.Logit(model_df["denied"], X_adj).fit(disp=False, maxiter=200)

# Odds ratios for gender effect
or_unadj = float(np.exp(model_unadj.params["female_ind"]))
or_adj = float(np.exp(model_adj.params["female_ind"]))

# Predicted denial at mean covariates
means = model_df[other_preds].mean()

def predict_denial(female_value: float) -> float:
    row = pd.Series({"const": 1.0, "female_ind": female_value})
    for p in means.index:
        row[p] = means[p]
    return float(model_adj.predict(row)[0])

pred_male = predict_denial(0.0)
pred_female = predict_denial(1.0)

# Output key results
print("Rows used:", len(model_df))
print("Denial rates by gender (mean):")
print(rates)
print("\nUnadjusted logit (denial ~ female):")
print(model_unadj.summary())
print("\nAdjusted logit (denial ~ female + controls):")
print(model_adj.summary())
print("\nOdds ratio (female) unadjusted:", or_unadj)
print("Odds ratio (female) adjusted:", or_adj)
print("Predicted denial at mean covariates (male, female):", pred_male, pred_female)
