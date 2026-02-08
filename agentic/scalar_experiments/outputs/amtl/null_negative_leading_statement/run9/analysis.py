import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

# Load data
_df = pd.read_csv("amtl.csv")

# Basic checks
_df = _df.copy()
_df = _df[_df["sockets"] > 0]
_df["amtl_prop"] = _df["num_amtl"] / _df["sockets"]

# Fit binomial GLM with frequency weights
formula = "amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)"
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df["sockets"],
).fit()

# Prepare settings for marginal predictions
age_mean = float(_df["age"].mean())
prob_male_mean = float(_df["prob_male"].mean())

# Tooth class distribution in data (overall)
tooth_weights = (
    _df["tooth_class"].value_counts(normalize=True).to_dict()
)

# Genus list
all_genera = sorted(_df["genus"].unique().tolist())

# Build prediction rows for each genus and tooth class
rows = []
for g in all_genera:
    for tc, w in tooth_weights.items():
        rows.append(
            {
                "genus": g,
                "tooth_class": tc,
                "age": age_mean,
                "prob_male": prob_male_mean,
                "weight": w,
            }
        )

pred_df = pd.DataFrame(rows)

# Use model to get design matrix that matches the fitted model
design_info = model.model.data.design_info
exog = patsy.build_design_matrices([design_info], pred_df)[0]

# Simulate coefficients to get uncertainty
coef = model.params.values
cov = model.cov_params().values
rng = np.random.default_rng(42)

n_sim = 20000
coef_sims = rng.multivariate_normal(coef, cov, size=n_sim)

# Compute weighted mean predicted probability per genus per sim
# Logistic function

def expit(x):
    return 1.0 / (1.0 + np.exp(-x))

# Map each row to its genus and weight
row_genus = pred_df["genus"].values
row_weight = pred_df["weight"].values

# Compute linear predictors for all sims
linpred = coef_sims @ exog.T  # shape (n_sim, n_rows)
probs = expit(linpred)

# Aggregate by genus
unique_genera = all_genera

# Precompute indices for each genus
genus_indices = {g: np.where(row_genus == g)[0] for g in unique_genera}

mean_prob = {}
for g in unique_genera:
    idx = genus_indices[g]
    w = row_weight[idx]
    # normalize weights (should sum to 1)
    w = w / w.sum()
    mean_prob[g] = (probs[:, idx] * w).sum(axis=1)

# Compare Homo sapiens to non-human genera
human_label = "Homo sapiens"
non_humans = [g for g in unique_genera if g != human_label]

# If label differs, fail fast
if human_label not in mean_prob:
    raise ValueError(f"Expected human label '{human_label}' not found in genus.")

human_probs = mean_prob[human_label]
nonhuman_mean = np.mean([mean_prob[g] for g in non_humans], axis=0)

# Probability that humans have higher AMTL than non-human average
p_higher = float(np.mean(human_probs > nonhuman_mean))

# Map to Likert scale [-100, 100]
score = int(np.round(200 * (p_higher - 0.5)))
score = max(-100, min(100, score))

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(score))

# Write a brief analysis summary for debugging (not required but helpful locally)
summary = {
    "p_higher": p_higher,
    "score": score,
    "mean_prob_human": float(np.mean(human_probs)),
    "mean_prob_nonhuman": float(np.mean(nonhuman_mean)),
    "genera": unique_genera,
}
with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
