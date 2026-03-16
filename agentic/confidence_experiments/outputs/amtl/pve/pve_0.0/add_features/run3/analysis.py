import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("amtl.csv")

# Keep relevant columns and drop rows with missing values in these columns
cols = ["num_amtl", "age", "prob_male", "tooth_class", "genus"]

df = df[cols].dropna()

# Ensure categorical types

df["tooth_class"] = df["tooth_class"].astype("category")
df["genus"] = df["genus"].astype("category")

# Set baseline genus to Papio for interpretability
if "Papio" in df["genus"].cat.categories:
    new_order = ["Papio"] + [c for c in df["genus"].cat.categories if c != "Papio"]
    df["genus"] = df["genus"].cat.reorder_categories(new_order, ordered=False)

# Fit OLS model with robust standard errors
model = smf.ols("num_amtl ~ C(genus) + age + prob_male + C(tooth_class)", data=df).fit(cov_type="HC3")

# Extract pairwise contrasts: Homo sapiens vs each non-human genus
# Build a contrast vector for parameters
params = model.params
param_names = list(params.index)

# Helper to build contrast

def contrast_vector(a, b):
    # contrast for mean difference between genus a and b
    # This uses coding: intercept is baseline (Papio). For non-baseline, coefficient is difference from baseline.
    # For baseline vs other, the difference is just the other coefficient (or its negative).
    v = np.zeros(len(param_names))
    # Identify coefficient names
    name_a = f"C(genus)[T.{a}]"
    name_b = f"C(genus)[T.{b}]"
    if a == "Papio":
        # Papio is baseline
        if name_b in param_names:
            v[param_names.index(name_b)] = -1.0
        else:
            raise ValueError(f"No coefficient for {b}")
    elif b == "Papio":
        if name_a in param_names:
            v[param_names.index(name_a)] = 1.0
        else:
            raise ValueError(f"No coefficient for {a}")
    else:
        if name_a in param_names:
            v[param_names.index(name_a)] = 1.0
        if name_b in param_names:
            v[param_names.index(name_b)] = -1.0
    return v

# Determine available genera

unique_genera = list(df["genus"].cat.categories)

# Compute contrasts for Homo sapiens vs each non-human genus
results = {}

for other in [g for g in unique_genera if g != "Homo sapiens"]:
    v = contrast_vector("Homo sapiens", other)
    ttest = model.t_test(v)
    results[other] = {
        "estimate": float(ttest.effect),
        "se": float(ttest.sd),
        "t": float(ttest.tvalue),
        "pvalue": float(ttest.pvalue),
    }

# Output results and model summary metrics

print("n_rows", len(df))
print("genera", unique_genera)
print("model_r2", model.rsquared)
print("results", results)

# Also compute adjusted mean predictions by genus at mean age/prob_male and average tooth_class distribution
mean_age = df["age"].mean()
mean_prob_male = df["prob_male"].mean()

# Use observed tooth_class distribution for marginalization

tooth_dist = df["tooth_class"].value_counts(normalize=True)

preds = {}
for genus in unique_genera:
    # Weighted average over tooth_class
    weighted_pred = 0.0
    for tc, w in tooth_dist.items():
        pred = model.predict(pd.DataFrame({
            "genus": [genus],
            "age": [mean_age],
            "prob_male": [mean_prob_male],
            "tooth_class": [tc],
        }))[0]
        weighted_pred += w * pred
    preds[genus] = float(weighted_pred)

print("adjusted_pred", preds)
