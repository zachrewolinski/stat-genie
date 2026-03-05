import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "amtl.csv"

df = pd.read_csv(path)

# Rename columns for clarity
rename_map = {
    "feature1": "tooth_class",
    "feature2": "specimen_id",
    "feature3": "missing_teeth",
    "feature4": "scorable_sockets",
    "feature5": "age",
    "feature6": "age_uncertainty",
    "feature7": "sex",
    "feature8": "genus",
    "feature9": "region",
}

df = df.rename(columns=rename_map)

# Drop rows with missing or zero scorable_sockets (can't compute proportion)
# Also drop rows with missing key covariates
needed = ["missing_teeth", "scorable_sockets", "age", "sex", "tooth_class", "genus"]

df = df.dropna(subset=needed)

df = df[df["scorable_sockets"] > 0].copy()

# Ensure categories
# Explicit reference for genus
if "Pan" in df["genus"].unique():
    genus_order = ["Pan", "Homo sapiens", "Papio", "Pongo"]
else:
    genus_order = sorted(df["genus"].unique())

df["genus"] = pd.Categorical(df["genus"], categories=genus_order)

df["tooth_class"] = pd.Categorical(df["tooth_class"], categories=sorted(df["tooth_class"].unique()))

# Response as proportion with frequency weights

df["missing_prop"] = df["missing_teeth"] / df["scorable_sockets"]

# Fit binomial GLM
formula = "missing_prop ~ C(genus, Treatment(reference='Pan')) + age + sex + C(tooth_class)"
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df["scorable_sockets"]).fit()

# Wald tests for Homo vs other genera
# Coef names: C(genus, Treatment(reference='Pan'))[T.Homo sapiens]

coef_names = model.params.index.tolist()

def wald_test_linear_combination(lincomb):
    # lincomb: dict of param->weight
    vec = np.zeros(len(coef_names))
    for k, v in lincomb.items():
        if k in coef_names:
            vec[coef_names.index(k)] = v
        else:
            raise ValueError(f"Parameter {k} not found")
    return model.t_test(vec)

homo_vs_pan = wald_test_linear_combination({"C(genus, Treatment(reference='Pan'))[T.Homo sapiens]": 1.0})

# Homo vs Papio: (Homo - Papio) = beta_homo - beta_papio
papio_param = "C(genus, Treatment(reference='Pan'))[T.Papio]"

if papio_param in coef_names:
    homo_vs_papio = wald_test_linear_combination({
        "C(genus, Treatment(reference='Pan'))[T.Homo sapiens]": 1.0,
        papio_param: -1.0,
    })
else:
    homo_vs_papio = None

pongo_param = "C(genus, Treatment(reference='Pan'))[T.Pongo]"
if pongo_param in coef_names:
    homo_vs_pongo = wald_test_linear_combination({
        "C(genus, Treatment(reference='Pan'))[T.Homo sapiens]": 1.0,
        pongo_param: -1.0,
    })
else:
    homo_vs_pongo = None

# Marginal standardized predicted probabilities by genus
# For each genus, set genus for all rows, keep covariates, predict, then average weighted by sockets

preds = {}
for g in df["genus"].cat.categories:
    tmp = df.copy()
    tmp["genus"] = g
    pred = model.predict(tmp)
    # weighted average by scorable sockets to reflect tooth counts
    preds[g] = np.average(pred, weights=tmp["scorable_sockets"])

# Build results summary
results = {
    "n_rows": int(df.shape[0]),
    "genus_levels": df["genus"].cat.categories.tolist(),
    "model_params": model.params.to_dict(),
    "model_pvalues": model.pvalues.to_dict(),
    "homo_vs_pan": {
        "coef": float(homo_vs_pan.effect[0]),
        "pvalue": float(homo_vs_pan.pvalue),
    },
    "homo_vs_papio": None,
    "homo_vs_pongo": None,
    "predicted_probabilities": {k: float(v) for k, v in preds.items()},
}

if homo_vs_papio is not None:
    results["homo_vs_papio"] = {
        "coef": float(homo_vs_papio.effect[0]),
        "pvalue": float(homo_vs_papio.pvalue),
    }

if homo_vs_pongo is not None:
    results["homo_vs_pongo"] = {
        "coef": float(homo_vs_pongo.effect[0]),
        "pvalue": float(homo_vs_pongo.pvalue),
    }

# Save results to json for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print concise summary
print("Rows:", results["n_rows"])
print("Genus levels:", results["genus_levels"])
print("Predicted probabilities:")
for k, v in results["predicted_probabilities"].items():
    print(f"  {k}: {v:.4f}")
print("Homo vs Pan: coef=%.4f, p=%.4g" % (results["homo_vs_pan"]["coef"], results["homo_vs_pan"]["pvalue"]))
if results["homo_vs_papio"]:
    print("Homo vs Papio: coef=%.4f, p=%.4g" % (results["homo_vs_papio"]["coef"], results["homo_vs_papio"]["pvalue"]))
if results["homo_vs_pongo"]:
    print("Homo vs Pongo: coef=%.4f, p=%.4g" % (results["homo_vs_pongo"]["coef"], results["homo_vs_pongo"]["pvalue"]))
