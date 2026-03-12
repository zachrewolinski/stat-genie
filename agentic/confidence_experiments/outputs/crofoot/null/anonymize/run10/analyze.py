import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Define variables
# Relative group size: focal size - other size
# Contest location advantage: other distance from its center minus focal distance from its center
# Positive loc_adv means contest is relatively closer to the focal group's center

df["rel_size"] = df["feature7"] - df["feature8"]
df["loc_adv"] = df["feature6"] - df["feature5"]

# Standardize predictors for comparability
for col in ["rel_size", "loc_adv"]:
    mean = df[col].mean()
    std = df[col].std(ddof=0)
    df[f"{col}_z"] = (df[col] - mean) / std if std != 0 else 0.0

# Outcome
Y = df["feature4"]

# Logistic regression with main effects
X = sm.add_constant(df[["rel_size_z", "loc_adv_z"]])
logit = sm.Logit(Y, X).fit(disp=False)

# Optional interaction model for context
X_int = sm.add_constant(df[["rel_size_z", "loc_adv_z"]])
X_int["interaction"] = df["rel_size_z"] * df["loc_adv_z"]
logit_int = sm.Logit(Y, X_int).fit(disp=False)

# Extract stats
summary = {
    "n": int(df.shape[0]),
    "rel_size_mean": float(df["rel_size"].mean()),
    "rel_size_sd": float(df["rel_size"].std(ddof=0)),
    "loc_adv_mean": float(df["loc_adv"].mean()),
    "loc_adv_sd": float(df["loc_adv"].std(ddof=0)),
}

params = logit.params
pvalues = logit.pvalues
conf = logit.conf_int()

results = {}
for var in ["rel_size_z", "loc_adv_z"]:
    beta = float(params[var])
    pval = float(pvalues[var])
    ci_low, ci_high = map(float, conf.loc[var])
    or_ = float(np.exp(beta))
    or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
    results[var] = {
        "beta": beta,
        "p": pval,
        "or": or_,
        "or_ci": or_ci,
    }

# Interaction p-value
interaction_p = float(logit_int.pvalues.get("interaction", np.nan))

# Determine qualitative evidence
rel_size_sig = results["rel_size_z"]["p"] < 0.05
loc_adv_sig = results["loc_adv_z"]["p"] < 0.05

# Build explanation
explanation_lines = []
explanation_lines.append(
    "Logistic regression predicting focal-group win (feature4) from relative group size (focal minus other) "
    "and contest location advantage (other distance from its center minus focal distance from its center; higher means contest is closer to focal group's center)."
)
explanation_lines.append(
    f"Sample size: n={summary['n']}. Predictors were standardized for the model."
)

for label, key in [("Relative group size", "rel_size_z"), ("Location advantage", "loc_adv_z")]:
    r = results[key]
    explanation_lines.append(
        f"{label}: beta={r['beta']:.3f}, OR={r['or']:.2f} (95% CI {r['or_ci'][0]:.2f}-{r['or_ci'][1]:.2f}), p={r['p']:.4f}."
    )

explanation_lines.append(
    f"Interaction (size x location) model check: p={interaction_p:.4f} (not used for the main conclusion)."
)

# Determine response score
# Heuristic: both significant -> strong yes; one significant -> moderate yes; none -> no.
if rel_size_sig and loc_adv_sig:
    response = 85
    conclusion = "Both relative group size and contest location show statistically significant effects on win probability."
elif rel_size_sig or loc_adv_sig:
    response = 65
    conclusion = "There is evidence that at least one of the two factors influences win probability, but support is not strong for both simultaneously."
else:
    response = 30
    conclusion = "There is no clear statistical evidence that relative group size and contest location influence win probability in this dataset."

explanation_lines.append(conclusion)

output = {
    "response": int(response),
    "explanation": " ".join(explanation_lines),
}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(output, f)

print(json.dumps(output, indent=2))
