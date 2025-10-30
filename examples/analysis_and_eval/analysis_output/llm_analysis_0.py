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

    # Keep only columns needed for modeling
    required_cols = [
        'masfem', 'gender_mf', 'alldeaths', 'ndam15',
        'wind', 'category', 'min', 'year', 'elapsedyrs'
    ]

    # Drop rows with missing values in any of the required columns
    df = df.dropna(subset=required_cols)

    # Standardize the continuous femininity rating for interpretability
    # (z-score: mean 0, sd 1). Keep original masfem as well if desired.
    df['masfem_std'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Ensure gender_mf is binary numeric (0/1)
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')

    # Create a log-transformed damage variable for a robustness check (highly skewed)
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Optionally coerce numeric controls to numeric types
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # After coercion, drop any rows that became NaN
    df = df.dropna(subset=['masfem_std', 'gender_mf', 'alldeaths', 'wind', 'category', 'min', 'year', 'elapsedyrs'])

    # Reset index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs two complementary models to test whether more-feminine hurricane names are associated
    with outcomes consistent with fewer precautionary actions by the public.

    1) Negative binomial regression predicting alldeaths (count outcome). Negative binomial
       handles overdispersion relative to Poisson.
    2) OLS regression predicting log-transformed economic damage (log_ndam15) as a robustness check.

    Both models include the same set of controls and a test interaction between name femininity
    and storm category (category as moderator of masfem effect).

    Returns a dict with the fitted results objects.
    """
    import statsmodels.api as sm

    results = {}

    # Define predictors and interaction term
    base_predictors = ['masfem_std', 'gender_mf', 'wind', 'category', 'min', 'year', 'elapsedyrs']

    # Prepare design matrix X
    X = df[base_predictors].copy()

    # Create interaction: masfem_std * category (center category to aid interpretation)
    X['category_c'] = X['category'] - X['category'].mean()
    X['masfem_x_category'] = X['masfem_std'] * X['category_c']

    # Final predictor set for the models
    predictors = ['masfem_std', 'gender_mf', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'masfem_x_category']

    X_model = X[predictors]
    X_model = sm.add_constant(X_model)

    # 1) Negative Binomial for counts of deaths
    y_counts = df['alldeaths'].astype(float)

    # Fit GLM Negative Binomial (uses a log link by default for counts)
    try:
        nb_model = sm.GLM(y_counts, X_model, family=sm.families.NegativeBinomial()).fit()
        results['neg_bin_alldeaths'] = nb_model
    except Exception as e:
        # If GLM NegativeBinomial fails, fall back to Poisson with robust SEs
        pois = sm.GLM(y_counts, X_model, family=sm.families.Poisson()).fit(cov_type='HC0')
        results['neg_bin_alldeaths'] = pois
        results['neg_bin_alldeaths_warning'] = f"NegativeBinomial failed; fallback to Poisson. Error: {e}"

    # 2) OLS for log-transformed damage (robustness)
    y_damage = df['log_ndam15'].astype(float)
    ols_model = sm.OLS(y_damage, X_model).fit(cov_type='HC1')  # robust SEs
    results['ols_log_ndam15'] = ols_model

    return results


