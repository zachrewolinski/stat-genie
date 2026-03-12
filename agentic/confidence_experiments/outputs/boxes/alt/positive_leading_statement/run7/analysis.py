import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DATA_PATH = Path("boxes.csv")
df = pd.read_csv(DATA_PATH)

# Basic sanity checks
summary = {
    "n": int(df.shape[0]),
    "y_counts": df["y"].value_counts().to_dict(),
    "age_range": (float(df["age"].min()), float(df["age"].max())),
    "cultures": sorted(df["culture"].unique().tolist()),
}

# Create derived variables
# Reliance on social information: choosing majority or minority (2 or 3) vs undemonstrated option (1)
df["social_choice"] = df["y"].isin([2, 3]).astype(int)

# Majority preference among social choices: majority (2) vs minority (3)
mask_social = df["y"].isin([2, 3])
df_social = df.loc[mask_social].copy()
df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

# Helper to build model summaries in a structured way
results = {
    "descriptives": summary,
    "models": {},
}

# 1) Overall reliance on social information
prop_social = df["social_choice"].mean()
results["reliance_overall"] = {
    "prop_social": float(prop_social),
}

# 2) Logistic regression: social_choice ~ age + gender + majority_first + C(culture)
try:
    model_reliance = smf.logit(
        "social_choice ~ age + gender + majority_first + C(culture)", data=df
    ).fit(disp=False)
    results["models"]["reliance_logit"] = {
        "llf": float(model_reliance.llf),
        "pseudo_r2": float(model_reliance.prsquared),
        "nobs": int(model_reliance.nobs),
        "params": model_reliance.params.to_dict(),
        "pvalues": model_reliance.pvalues.to_dict(),
    }
except Exception as e:  # pragma: no cover - robustness
    results["models"]["reliance_logit_error"] = str(e)

# 3) Logistic regression among social choices: majority_choice ~ age + gender + majority_first + C(culture)
try:
    model_majority = smf.logit(
        "majority_choice ~ age + gender + majority_first + C(culture)",
        data=df_social,
    ).fit(disp=False)
    results["models"]["majority_logit"] = {
        "llf": float(model_majority.llf),
        "pseudo_r2": float(model_majority.prsquared),
        "nobs": int(model_majority.nobs),
        "params": model_majority.params.to_dict(),
        "pvalues": model_majority.pvalues.to_dict(),
    }
except Exception as e:  # pragma: no cover
    results["models"]["majority_logit_error"] = str(e)

# 4) Multinomial model for full three-way choice: y ~ age + gender + majority_first + C(culture)
try:
    # Recode y as categorical starting at 0 for MNLogit
    df_mn = df.copy()
    df_mn["y_cat"] = df_mn["y"] - 1
    mn_endog = df_mn["y_cat"]
    # Use treatment coding for culture
    mn_exog = pd.get_dummies(df_mn[["age", "gender", "majority_first", "culture"]], columns=["culture"], drop_first=True)
    mn_exog = sm.add_constant(mn_exog, has_constant="add")
    mn_model = sm.MNLogit(mn_endog, mn_exog).fit(disp=False)
    results["models"]["multinomial"] = {
        "llf": float(mn_model.llf),
        "pseudo_r2": float(mn_model.prsquared),
        "nobs": int(mn_model.nobs),
        # Store p-values for each predictor across comparisons
        "pvalues": {k: float(v) for k, v in mn_model.pvalues.mean(axis=0).items()},
    }
except Exception as e:  # pragma: no cover
    results["models"]["multinomial_error"] = str(e)

# Save full numeric results for inspection
Path("analysis_results.json").write_text(json.dumps(results, indent=2))

