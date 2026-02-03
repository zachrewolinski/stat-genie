import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF_PATH = "affairs.csv"
df = pd.read_csv(DF_PATH)

# Map key variables
# feature2: frequency of extramarital intercourse in past year
# feature6: children in marriage (yes/no)

df = df.copy()

df["affairs"] = df["feature2"]
df["children"] = df["feature6"].str.lower().str.strip()

# Binary outcome: any affairs

df["any_affair"] = (df["affairs"] > 0).astype(int)

# Descriptive stats by children

desc = df.groupby("children").agg(
    n=("affairs", "size"),
    mean_affairs=("affairs", "mean"),
    median_affairs=("affairs", "median"),
    prop_any_affair=("any_affair", "mean"),
)

# Prepare regression design matrix
# Controls: gender (feature3), age (feature4), years married (feature5),
# religiosity (feature7), education (feature8), occupation (feature9),
# marriage rating (feature10)

model_df = df[[
    "any_affair",
    "affairs",
    "children",
    "feature3",
    "feature4",
    "feature5",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
]].copy()

# Encode categorical variables
model_df = pd.get_dummies(model_df, columns=["children", "feature3"], drop_first=True)

# Ensure numeric
for col in model_df.columns:
    model_df[col] = pd.to_numeric(model_df[col], errors="ignore")

# Logistic regression for any affair
X_logit = model_df.drop(columns=["any_affair", "affairs"])
X_logit = sm.add_constant(X_logit, has_constant="add")

y_logit = model_df["any_affair"]

logit_result = None
try:
    logit_model = sm.Logit(y_logit, X_logit)
    logit_result = logit_model.fit(disp=False)
except Exception:
    # Fallback to GLM binomial if Logit fails to converge
    glm_model = sm.GLM(y_logit, X_logit, family=sm.families.Binomial())
    logit_result = glm_model.fit()

# OLS regression for affairs frequency (continuous)
X_ols = model_df.drop(columns=["any_affair", "affairs"])
X_ols = sm.add_constant(X_ols, has_constant="add")

y_ols = model_df["affairs"]

ols_model = sm.OLS(y_ols, X_ols)
ols_result = ols_model.fit(cov_type="HC3")

# Extract children coefficient (children_yes)
child_col = [c for c in X_logit.columns if c.startswith("children_")]
child_col = child_col[0] if child_col else None

results_summary = {
    "desc": desc,
    "logit_child_coef": None,
    "logit_child_p": None,
    "ols_child_coef": None,
    "ols_child_p": None,
}

if child_col:
    results_summary["logit_child_coef"] = float(logit_result.params.get(child_col, np.nan))
    results_summary["logit_child_p"] = float(logit_result.pvalues.get(child_col, np.nan))
    results_summary["ols_child_coef"] = float(ols_result.params.get(child_col, np.nan))
    results_summary["ols_child_p"] = float(ols_result.pvalues.get(child_col, np.nan))

print("Descriptives by children:\n", desc)
print("\nLogit child coef, p:", results_summary["logit_child_coef"], results_summary["logit_child_p"])
print("OLS child coef, p:", results_summary["ols_child_coef"], results_summary["ols_child_p"])
