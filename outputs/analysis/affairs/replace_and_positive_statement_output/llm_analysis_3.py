from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_and_positive_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop rows with missing key variables
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Dependent variable: keep 'affairs' as numeric count
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Independent variable: children -> binary 1=yes, 0=no
    df['Children'] = df['children'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Gender indicator: 1 if male, 0 if female (map defensively)
    df['Gender_Male'] = df['gender'].astype(str).str.lower().map({'male': 1, 'female': 0})

    # Rename and ensure numeric for continuous controls
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows created as NaN by conversions
    df = df.dropna(subset=['affairs', 'Children', 'Gender_Male', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating'])

    # Standardize continuous controls (z-scores) to improve numerical stability and interpretation
    for col, newcol in [('Age', 'Age_z'), ('YearsMarried', 'YearsMarried_z'), ('Religiousness', 'Religiousness_z'),
                        ('Education', 'Education_z'), ('Occupation', 'Occupation_z'), ('Rating', 'Rating_z')]:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            # fallback to zero if no variation (should be rare)
            df[newcol] = 0.0
        else:
            df[newcol] = (df[col] - mean) / std

    # Keep only the columns necessary for modeling (but return full df so user can inspect)
    # Final dataframe contains at minimum the columns listed in the conceptual variables
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count models to estimate effect of having children on frequency of affairs.
    Returns both a Poisson model (baseline) and a Negative Binomial model (accounts for overdispersion),
    plus an IRR table and a simple overdispersion diagnostic.
    """
    df = df.copy()

    # Columns used in the model (must match transform output)
    exog_cols = ['Children', 'Gender_Male', 'Age_z', 'YearsMarried_z', 'Religiousness_z', 'Education_z', 'Occupation_z', 'Rating_z']

    # Ensure columns exist
    missing = [c for c in exog_cols + ['affairs'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    X = sm.add_constant(df[exog_cols])
    y = df['affairs']

    # Fit Poisson (baseline)
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type='HC0')

    # Check overdispersion: deviance / df_resid should be ~1 under Poisson
    overdispersion = None
    try:
        overdispersion = poisson_model.deviance / poisson_model.df_resid
    except Exception:
        overdispersion = float('nan')

    # Fit Negative Binomial to account for overdispersion
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(cov_type='HC0')

    # Compute incidence rate ratios (IRRs) and 95% CIs from NB model
    nb_params = nb_model.params
    try:
        nb_conf = nb_model.conf_int()
    except Exception:
        nb_conf = nb_model.conf_int()
    irr = pd.DataFrame({
        'coef': nb_params,
        'IRR': np.exp(nb_params),
        'IRR_lower': np.exp(nb_conf[0]),
        'IRR_upper': np.exp(nb_conf[1])
    })

    # Prepare a concise summary of the coefficient on Children
    children_coef_poisson = poisson_model.params.get('Children', np.nan)
    children_pval_poisson = poisson_model.pvalues.get('Children', np.nan)
    children_coef_nb = nb_model.params.get('Children', np.nan)
    children_pval_nb = nb_model.pvalues.get('Children', np.nan)

    summary = {
        'children_coef_poisson': float(children_coef_poisson),
        'children_pval_poisson': float(children_pval_poisson),
        'children_coef_nb': float(children_coef_nb),
        'children_pval_nb': float(children_pval_nb),
        'overdispersion_poisson_deviance_per_df': float(overdispersion)
    }

    results = {
        'poisson_model': poisson_model,
        'nb_model': nb_model,
        'nb_irrs': irr,
        'overdispersion': overdispersion,
        'children_summary': summary
    }

    return results


