from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying the original
    df = df.copy()

    # ----------------------
    # Required columns and basic cleaning
    # ----------------------
    # Keep rows with non-missing dependent variable and main IVs
    required_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'source']
    df = df.dropna(subset=required_cols)

    # Rename 'min' to avoid collision with Python built-in; keep original values
    df['min_pressure'] = df['min']

    # Ensure numeric types where appropriate
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(int)
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['min_pressure'] = pd.to_numeric(df['min_pressure'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min_pressure', 'year', 'elapsedyrs', 'source'])

    # ----------------------
    # Derived / transformed columns
    # ----------------------
    # Standardize masfem so coefficients are interpretable (z-score)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Keep alldeaths as count for count modeling; also create log version for diagnostics/robustness
    df['alldeaths_log1p'] = np.log1p(df['alldeaths'])

    # Make sure 'source' is a categorical variable for modeling with C(source)
    df['source'] = df['source'].astype('category')

    # Optional: create an indicator for extremely high-fatality outliers for sensitivity checks
    df['alldeaths_outlier'] = (df['alldeaths'] > df['alldeaths'].quantile(0.99)).astype(int)

    # Finalize: keep only columns that will be used in the model and useful diagnostics
    keep_cols = ['alldeaths', 'alldeaths_log1p', 'masfem', 'masfem_z', 'gender_mf', 'wind', 'category', 'min_pressure', 'year', 'elapsedyrs', 'source', 'alldeaths_outlier']
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Build formula: negative binomial GLM with categorical source control
    formula = 'alldeaths ~ masfem_z + gender_mf + wind + category + min_pressure + year + elapsedyrs + C(source)'

    # Fit Negative Binomial GLM to account for over-dispersed count data
    # Using statsmodels' GLM with NegativeBinomial family
    nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
    results = nb_model.fit()

    # Return the fitted results object. The caller can call .summary() or inspect params, conf_int, etc.
    return results


