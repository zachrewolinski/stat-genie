import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DATA_PATH = "amtl.csv"
df = pd.read_csv(DATA_PATH)

# Prepare variables
# Binary indicator for modern humans vs non-human primates

df["human"] = (df["genus"] == "Homo sapiens").astype(int)

# Fit binomial regression: counts of AMTL out of sockets
# Use a two-column endogenous array: [successes, failures]
endog = np.column_stack([df["num_amtl"], df["sockets"] - df["num_amtl"]])

# Design matrix with categorical tooth_class
exog = pd.get_dummies(
    df[["human", "age", "prob_male", "tooth_class"]],
    columns=["tooth_class"],
    drop_first=True,
)
exog = sm.add_constant(exog, has_constant="add")

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Extract effect for humans
coef = res.params.get("human", np.nan)
pval = res.pvalues.get("human", np.nan)

# Compute odds ratio
odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Compute marginal predicted AMTL rate for human vs non-human
# Use mean age and prob_male, and average across tooth classes weighted by their frequency
mean_age = df["age"].mean()
mean_prob_male = df["prob_male"].mean()

# Weight by tooth_class distribution
class_weights = df["tooth_class"].value_counts(normalize=True)

preds = {}
for human in [0, 1]:
    # Build a small prediction dataframe for each tooth class
    pred_rows = []
    for tc, w in class_weights.items():
        pred_rows.append({
            "human": human,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": tc,
            "weight": w,
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_exog = pd.get_dummies(
        pred_df[["human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    )
    # Align columns with training exog
    pred_exog = pred_exog.reindex(columns=exog.columns, fill_value=0)
    pred_df["pred"] = res.predict(pred_exog)
    preds[human] = float((pred_df["pred"] * pred_df["weight"]).sum())

# Decide conclusion
# "Yes" if humans have higher AMTL odds (positive coef) and statistically significant
alpha = 0.05
answer_yes = bool(np.isfinite(coef) and coef > 0 and pval < alpha)

# Write conclusion
conclusion_lines = []
conclusion_lines.append("Yes" if answer_yes else "No")

if answer_yes:
    reasoning = (
        f"Controlling for age, sex probability, and tooth class, the human indicator has a positive "
        f"effect (odds ratio {odds_ratio:.2f}, p={pval:.3g}). "
        f"The model predicts a higher AMTL rate for humans (~{preds[1]:.3f}) than non-humans (~{preds[0]:.3f})."
    )
else:
    if np.isfinite(coef):
        direction = "positive" if coef > 0 else "negative"
        reasoning = (
            f"After controlling for age, sex probability, and tooth class, the human effect is {direction} "
            f"(odds ratio {odds_ratio:.2f}, p={pval:.3g}), which does not support higher AMTL in humans. "
            f"Predicted AMTL rates are ~{preds[1]:.3f} for humans and ~{preds[0]:.3f} for non-humans."
        )
    else:
        reasoning = "The model did not estimate a reliable human effect after controlling for covariates."

conclusion_lines.append(reasoning)

with open("conclusion.txt", "w") as f:
    f.write("\n".join(conclusion_lines) + "\n")

# Save a brief text report for inspection
with open("analysis_report.txt", "w") as f:
    f.write(res.summary().as_text())
    f.write("\n\n")
    f.write(f"Human odds ratio: {odds_ratio:.4f}\n")
    f.write(f"Human p-value: {pval:.6g}\n")
    f.write(f"Predicted AMTL rate (non-human): {preds[0]:.6f}\n")
    f.write(f"Predicted AMTL rate (human): {preds[1]:.6f}\n")

print(res.summary())
print("Human odds ratio:", odds_ratio)
print("Human p-value:", pval)
print("Predicted AMTL rate (non-human):", preds[0])
print("Predicted AMTL rate (human):", preds[1])
