import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "amtl.csv"
df = pd.read_csv(DATA_PATH)

# Ensure categorical ordering for genus with Homo sapiens as baseline
if "genus" in df.columns:
    df["genus"] = pd.Categorical(
        df["genus"],
        categories=["Homo sapiens", "Pan", "Papio", "Pongo"],
        ordered=False,
    )

# Ensure tooth_class categorical
if "tooth_class" in df.columns:
    df["tooth_class"] = df["tooth_class"].astype("category")

# Primary model: genus-specific effects with controls
formula = "num_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
model = smf.ols(formula, data=df)
res = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": df["specimen"]},
)

# Secondary model: human vs non-human
if "genus" in df.columns:
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

formula_bin = "num_amtl ~ is_human + age + prob_male + C(tooth_class)"
model_bin = smf.ols(formula_bin, data=df)
res_bin = model_bin.fit(
    cov_type="cluster",
    cov_kwds={"groups": df["specimen"]},
)

# Extract genus coefficients (differences vs Homo sapiens)
genus_rows = {}
for genus in ["Pan", "Papio", "Pongo"]:
    term = f"C(genus)[T.{genus}]"
    if term in res.params.index:
        genus_rows[genus] = {
            "coef": float(res.params[term]),
            "p": float(res.pvalues[term]),
        }

# Extract binary human effect
if "is_human" in res_bin.params.index:
    human_effect = {
        "coef": float(res_bin.params["is_human"]),
        "p": float(res_bin.pvalues["is_human"]),
    }
else:
    human_effect = {"coef": float("nan"), "p": float("nan")}

# Compute marginal means by genus at mean covariates and averaged tooth_class
mean_age = float(df["age"].mean())
mean_prob_male = float(df["prob_male"].mean())

# Average over observed tooth_class distribution (to avoid arbitrary choice)
class_probs = df["tooth_class"].value_counts(normalize=True)

def marginal_mean_for_genus(genus_value: str) -> float:
    rows = []
    for tc, prob in class_probs.items():
        rows.append(
            {
                "genus": genus_value,
                "age": mean_age,
                "prob_male": mean_prob_male,
                "tooth_class": tc,
                "weight": prob,
            }
        )
    tmp = pd.DataFrame(rows)
    preds = res.predict(tmp)
    return float(np.sum(preds * tmp["weight"]))

marginal_means = {
    genus: marginal_mean_for_genus(genus)
    for genus in ["Homo sapiens", "Pan", "Papio", "Pongo"]
}

# Decision logic for Likert response
# Start from binary human effect and adjust based on genus-specific consistency
coef = human_effect["coef"]
pval = human_effect["p"]

# Strength based on effect size (standardized outcome) and significance
if np.isnan(coef) or np.isnan(pval):
    response = 50
    verdict = "inconclusive"
else:
    if coef <= 0:
        # Humans not higher
        response = 20 if pval < 0.05 else 40
        verdict = "no"
    else:
        # Humans higher
        if pval < 0.001:
            base = 85
        elif pval < 0.01:
            base = 75
        elif pval < 0.05:
            base = 65
        else:
            base = 55

        # Adjust for magnitude (in SD units)
        if coef >= 0.5:
            base += 5
        elif coef >= 0.3:
            base += 0
        elif coef >= 0.2:
            base -= 5
        else:
            base -= 10

        # Penalize if any genus-specific comparisons are not lower than humans
        inconsistent = 0
        nonsig = 0
        for g, stats in genus_rows.items():
            if stats["coef"] >= 0:
                inconsistent += 1
            if stats["p"] >= 0.05:
                nonsig += 1
        if inconsistent > 0:
            base -= 15
        elif nonsig > 0:
            base -= 5

        response = int(max(0, min(100, round(base))))
        verdict = "yes"

# Build explanation
lines = []
lines.append(
    "Fit clustered-robust OLS models predicting standardized AMTL (num_amtl) with controls for age, sex (prob_male), and tooth class, accounting for repeated specimens via cluster-robust SEs."
)
lines.append(
    f"Binary human vs non-human effect: coef = {coef:.3f}, p = {pval:.4f} (positive means higher AMTL in humans)."
)

if genus_rows:
    parts = []
    for g, stats in genus_rows.items():
        parts.append(f"{g} vs Homo: coef = {stats['coef']:.3f}, p = {stats['p']:.4f}")
    lines.append("Genus-specific contrasts: " + "; ".join(parts) + ".")

lines.append(
    "Marginal means (averaged over observed tooth-class distribution at mean age/sex): "
    + ", ".join([f"{g} = {marginal_means[g]:.3f}" for g in ["Homo sapiens", "Pan", "Papio", "Pongo"]])
    + "."
)

if verdict == "yes":
    lines.append(
        "Overall, humans show higher AMTL than non-human genera after adjustment; strength reflects effect size and statistical significance, with penalties if any genus-specific contrast is weak or inconsistent."
    )
elif verdict == "no":
    lines.append(
        "Overall, there is insufficient evidence that humans have higher AMTL after adjustment; the estimated difference is non-positive or not statistically significant."
    )
else:
    lines.append("Evidence is inconclusive given model outputs.")

explanation = " ".join(lines)

output = {"response": response, "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(output, f)
