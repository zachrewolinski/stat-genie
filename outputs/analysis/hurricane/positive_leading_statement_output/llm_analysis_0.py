from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/positive_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into an analysis-ready dataframe.

    Steps:
    - Drop rows missing core variables required for the primary analyses.
    - Standardize continuous name-femininity measures (masfem and masfem_mturk).
    - Create a logged damage variable for robustness checks (log_ndam15).
    - Center year to aid interpretation.
    - Create categorical dummy variables for category and source (drop_first=True).

    Returns the transformed dataframe containing all columns referenced in the model code.
    """
    df = df.copy()

    # Ensure expected columns exist
    required = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source', 'gender_mf', 'ndam15']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows missing core variables used in main models
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source'])

    # Standardize masfem (primary IV) and masfem_mturk (alternative IV)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        # create column of NAs if missing (keeps API consistent)
        df['masfem_mturk_z'] = np.nan

    # Create logged damage measure for robustness (clip to >=1 to avoid -inf for zeros)
    df['log_ndam15'] = np.log(df['ndam15'].clip(lower=1.0))

    # Center year
    df['year_c'] = df['year'] - df['year'].mean()

    # Ensure category and source are treated as categorical then create dummies (drop first to avoid multicollinearity)
    df['category'] = df['category'].astype(int)
    cat_dummies = pd.get_dummies(df['category'].astype(str), prefix='category', drop_first=True)
    # The dataset 'source' field contains strings; create dummies
    src_dummies = pd.get_dummies(df['source'].astype(str), prefix='source', drop_first=True)

    # Concatenate dummies to df
    df = pd.concat([df, cat_dummies, src_dummies], axis=1)

    # If some expected dummy columns are missing because a level is absent, ensure the column names still exist (fill with 0)
    # Expected category dummies: category_2..category_5 (since categories range 1..5). If any are missing, create them.
    for lev in ['category_2', 'category_3', 'category_4', 'category_5']:
        if lev not in df.columns:
            df[lev] = 0

    # For source dummies we do not know exact levels; but to be explicit, if standard names exist create them; otherwise, keep whatever dummies exist.
    # (Model code will pick up available source_* dummies.)

    # Keep only columns needed for modeling + the raw variables for transparency
    model_cols = [
        'alldeaths', 'masfem_z', 'gender_mf', 'wind', 'min', 'year_c', 'elapsedyrs',
        'masfem_mturk_z', 'log_ndam15',
        'category_2', 'category_3', 'category_4', 'category_5'
    ]

    # Also include any source dummies present
    src_cols = [c for c in df.columns if c.startswith('source_')]
    model_cols += src_cols

    # Return dataframe with model columns plus original important fields for auditing
    keep = list(dict.fromkeys(model_cols + ['name', 'ind']))  # preserve order and uniqueness
    df = df[keep]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to test whether more feminine hurricane names are associated
    with the outcome (alldeaths) after controlling for storm intensity and other covariates.

    Models fitted:
    1) Primary: Negative Binomial GLM predicting alldeaths from masfem_z + controls.
    2) Robustness A: Same as (1) but with gender_mf (binary) replacing masfem_z.
    3) Robustness B: OLS predicting log_ndam15 (logged damage) from masfem_z + controls.

    Returns a dictionary with fitted model results objects.
    """
    results = {}

    # Copy to avoid modifying caller's df
    dfm = df.copy()

    # Build the predictors list used in all models
    base_predictors = ['wind', 'min', 'year_c', 'elapsedyrs', 'masfem_z']

    # Add category dummies if present
    cat_preds = [c for c in ['category_2', 'category_3', 'category_4', 'category_5'] if c in dfm.columns]
    base_predictors += cat_preds

    # Add any source_* dummies present
    src_preds = [c for c in dfm.columns if c.startswith('source_')]
    base_predictors += src_preds

    # Ensure no missing values in predictors for rows used
    required_for_nb = ['alldeaths'] + base_predictors
    df_nb = dfm.dropna(subset=required_for_nb)

    # Prepare design matrix
    X_nb = df_nb[base_predictors].astype(float)
    X_nb = sm.add_constant(X_nb)
    y_nb = df_nb['alldeaths'].astype(float)

    # Fit Negative Binomial GLM (GLM NB handles overdispersion vs Poisson)
    try:
        nb_model = sm.GLM(y_nb, X_nb, family=sm.families.NegativeBinomial()).fit()
        results['nb_model'] = nb_model
    except Exception as e:
        results['nb_model_error'] = str(e)

    # Robustness A: binary gender_mf replacing masfem_z
    if 'gender_mf' in dfm.columns:
        preds_gender = [p for p in base_predictors if p != 'masfem_z'] + ['gender_mf']
        df_g = dfm.dropna(subset=['alldeaths'] + preds_gender)
        X_g = df_g[preds_gender].astype(float)
        X_g = sm.add_constant(X_g)
        y_g = df_g['alldeaths'].astype(float)
        try:
            nb_gender = sm.GLM(y_g, X_g, family=sm.families.NegativeBinomial()).fit()
            results['nb_model_gendermf'] = nb_gender
        except Exception as e:
            results['nb_model_gendermf_error'] = str(e)

    # Robustness B: OLS on logged damage (log_ndam15) as alternative outcome
    if 'log_ndam15' in dfm.columns:
        preds_damage = [p for p in base_predictors if p != 'masfem_z'] + ['masfem_z']
        df_d = dfm.dropna(subset=['log_ndam15'] + preds_damage)
        X_d = df_d[preds_damage].astype(float)
        X_d = sm.add_constant(X_d)
        y_d = df_d['log_ndam15'].astype(float)
        try:
            ols_damage = sm.OLS(y_d, X_d).fit()
            results['ols_damage'] = ols_damage
        except Exception as e:
            results['ols_damage_error'] = str(e)

    # Additional robustness: use masfem_mturk_z (if available and not all-NaN)
    if 'masfem_mturk_z' in dfm.columns and not dfm['masfem_mturk_z'].isna().all():
        preds_mturk = [p for p in base_predictors if p != 'masfem_z'] + ['masfem_mturk_z']
        df_m = dfm.dropna(subset=['alldeaths'] + preds_mturk)
        X_m = df_m[preds_mturk].astype(float)
        X_m = sm.add_constant(X_m)
        y_m = df_m['alldeaths'].astype(float)
        try:
            nb_mturk = sm.GLM(y_m, X_m, family=sm.families.NegativeBinomial()).fit()
            results['nb_model_mturk'] = nb_mturk
        except Exception as e:
            results['nb_model_mturk_error'] = str(e)

    # Return results dict. Each fitted model is a statsmodels results object with .summary(), params, etc.
    return results


