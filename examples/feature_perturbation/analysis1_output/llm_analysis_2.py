from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure expected columns exist; if some optional columns are missing we'll continue without them
    required_cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'gender_mf']

    # Drop rows missing the core variables used in the primary analysis
    core_required = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year']
    df = df.dropna(subset=core_required)

    # Create integer count outcome for deaths
    # Some datasets may already have integer type; coerce to int after filling or dropping NA rows above
    df['alldeaths_count'] = df['alldeaths'].astype(int)

    # Standardize masfem (z-score) to ease interpretation and numerical stability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # If masfem_mturk exists, create standardized version for robustness checks
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        # Ensure column exists (NaN) so downstream code can reference it safely
        df['masfem_mturk_z'] = np.nan

    # Center year to improve interpretability (reduces collinearity with intercept)
    df['year_c'] = df['year'] - df['year'].mean()

    # Make sure numeric controls are numeric (coerce if necessary)
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Ensure gender_mf is integer 0/1 if present
    if 'gender_mf' in df.columns:
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').fillna(0).astype(int)
    else:
        df['gender_mf'] = 0

    # Drop rows that became NA after coercion for the principal analysis variables
    df = df.dropna(subset=['alldeaths_count', 'masfem_z', 'wind', 'min', 'category', 'year_c'])

    # Return transformed dataframe with all columns used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Primary model: Negative Binomial regression for count outcome (alldeaths_count)
    # Rationale: alldeaths is a count variable and typically overdispersed relative to Poisson.

    # Build formula. We include masfem_z (standardized femininity) as the independent variable
    # and control for storm intensity (wind, min, category), temporal trend (year_c), elapsedyrs, and gender_mf.
    formula = 'alldeaths_count ~ masfem_z + wind + min + category + year_c + elapsedyrs + gender_mf'

    # Fit GLM with Negative Binomial family
    model = sm.GLM.from_formula(formula, data=df, family=sm.families.NegativeBinomial())
    results = model.fit()

    # Print summary for inspection and return the fitted results object
    print(results.summary())
    return results


