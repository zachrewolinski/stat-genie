import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "amtl.csv"
df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure no zero sockets (shouldn't happen) and drop missing essentials
_df = df.copy()
_df = _df[_df["sockets"] > 0]
_df = _df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

# Create human indicator
_df["human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Binomial GLM: proportion of AMTL with total trials = sockets
_df["amtl_rate"] = _df["num_amtl"] / _df["sockets"]

model = smf.glm(
    "amtl_rate ~ human + age + prob_male + C(tooth_class)",
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df["sockets"],
).fit()

# Extract coefficient for human indicator
coef = model.params["human"]
se = model.bse["human"]
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se
pval = model.pvalues["human"]

# Compute predicted rates for human vs nonhuman at mean covariates
mean_age = _df["age"].mean()
mean_prob_male = _df["prob_male"].mean()
# Choose reference tooth_class = first category in design (statsmodels uses alphabetical by default)
# We'll construct two rows with same covariates
pred_df = pd.DataFrame({
    "human": [0, 1],
    "age": [mean_age, mean_age],
    "prob_male": [mean_prob_male, mean_prob_male],
    "tooth_class": [_df["tooth_class"].astype(str).sort_values().iloc[0]] * 2,
})

pred_rates = model.predict(pred_df)

print("Human coefficient (log-odds):", coef)
print("SE:", se)
print("95% CI:", (ci_low, ci_high))
print("p-value:", pval)
print("Predicted AMTL rate (nonhuman, human) at mean covariates:", pred_rates.values)

# Save key results for conclusion
results = {
    "coef": coef,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "pval": pval,
    "pred_nonhuman": float(pred_rates.values[0]),
    "pred_human": float(pred_rates.values[1]),
}

pd.Series(results).to_json("analysis_results.json")
