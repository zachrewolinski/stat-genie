import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Rename for clarity
_df = _df.rename(
    columns={
        "feature1": "tooth_class",
        "feature2": "specimen_id",
        "feature3": "missing",
        "feature4": "total",
        "feature5": "age",
        "feature6": "age_uncertainty",
        "feature7": "sex",
        "feature8": "genus",
        "feature9": "region",
    }
)

# Basic cleaning
_df = _df.dropna(subset=["missing", "total", "age", "sex", "genus", "tooth_class"]).copy()
_df = _df[_df["total"] > 0].copy()
_df["present"] = _df["total"] - _df["missing"]

# Ensure categorical
_df["genus"] = _df["genus"].astype("category")
_df["tooth_class"] = _df["tooth_class"].astype("category")

# Set Homo sapiens as reference level
if "Homo sapiens" in _df["genus"].cat.categories:
    _df["genus"] = _df["genus"].cat.reorder_categories(
        ["Homo sapiens"] + [g for g in _df["genus"].cat.categories if g != "Homo sapiens"],
        ordered=False,
    )

# GLM binomial with counts
# Use two-column endog for missing/present
endog = _df[["missing", "present"]]

formula = "missing + present ~ C(genus) + age + sex + C(tooth_class)"

# Use GLM with Binomial
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial()).fit()

# Extract genus coefficients vs Homo sapiens
coef_table = model.summary2().tables[1].copy()

# Collect genus effects
genus_effects = {}
for g in _df["genus"].cat.categories:
    if g == "Homo sapiens":
        continue
    term = f"C(genus)[T.{g}]"
    if term in coef_table.index:
        genus_effects[g] = {
            "coef": coef_table.loc[term, "Coef."],
            "pval": coef_table.loc[term, "P>|z|"],
        }

# Predicted mean AMTL rate by genus at average covariates
mean_age = _df["age"].mean()
mean_sex = _df["sex"].mean()
ref_tooth = _df["tooth_class"].cat.categories[0]

pred_rows = []
for g in _df["genus"].cat.categories:
    pred_rows.append({"genus": g, "age": mean_age, "sex": mean_sex, "tooth_class": ref_tooth})

pred_df = pd.DataFrame(pred_rows)

pred = model.get_prediction(pred_df).summary_frame()

pred_df["predicted_rate"] = pred["mean"]

# Save key outputs for inspection
coef_table.to_csv("glm_coefficients.csv")
pred_df.to_csv("predicted_rates.csv", index=False)

# Print concise outputs
print("Genus effects vs Homo sapiens:")
for g, info in genus_effects.items():
    print(g, info)

print("\nPredicted AMTL rates by genus (at mean age/sex, ref tooth class):")
print(pred_df)

# Decision rule: Yes if all non-human genera have significantly lower odds (coef < 0 and p < 0.05)
all_lower_sig = True
for g, info in genus_effects.items():
    if not (info["coef"] < 0 and info["pval"] < 0.05):
        all_lower_sig = False
        break

with open("decision.txt", "w", encoding="utf-8") as f:
    f.write("Yes" if all_lower_sig else "No")
