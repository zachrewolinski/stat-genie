from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw input dataframe (feature1..feature14) into a cleaned dataframe containing all
    variables used in the statistical models. The function renames columns to descriptive names,
    drops rows with missing key fields, creates standardized and derived variables, and ensures
    correct dtypes for categorical variables.

    Final dataframe will include at minimum the columns referenced in the conceptual model:
      - fatalities (feature8)
      - masfem_z (z-scored feature4)
      - female_name (feature6 -> int 0/1)
      - max_wind (feature13)
      - min_pressure (feature5)
      - category (feature7 as categorical)
      - year_centered (feature2 centered)
      - log_damage2015 (log1p of feature14)
      - plus some descriptive columns (id, name, source, mturk_masfem) kept for diagnostics
    """
    # Rename raw feature columns to meaningful names
    df = df.rename(columns={
        'feature1': 'id',
        'feature2': 'year',
        'feature3': 'name',
        'feature4': 'masfem',
        'feature5': 'min_pressure',
        'feature6': 'female_name',
        'feature7': 'category',
        'feature8': 'fatalities',
        'feature9': 'damage_2013',
        'feature10': 'years_since',
        'feature11': 'source',
        'feature12': 'mturk_masfem',
        'feature13': 'max_wind',
        'feature14': 'damage_2015'
    })

    # Keep a copy of raw counts for diagnostics
    # Ensure numeric columns are numeric
    numeric_cols = ['masfem', 'min_pressure', 'female_name', 'category', 'fatalities', 'damage_2013', 'years_since', 'max_wind', 'damage_2015', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing key variables necessary for modeling
    df = df.dropna(subset=['masfem', 'fatalities', 'max_wind', 'min_pressure', 'category', 'year'])

    # Standardize/transform variables
    # z-score masfem (use population sd ddof=0 for stable scaling)
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) > 0 else 1.0
    df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std

    # Ensure female_name is integer 0/1
    df['female_name'] = df['female_name'].astype(int)

    # Ensure category is categorical
    df['category'] = df['category'].astype('category')

    # Create centered year to capture linear time trends
    df['year_centered'] = df['year'] - df['year'].mean()

    # Log-transform damage to reduce skew; use damage_2015 (normalized) if available
    df['damage_2015'] = pd.to_numeric(df['damage_2015'], errors='coerce')
    df['log_damage2015'] = np.log1p(df['damage_2015'].fillna(0.0))

    # Ensure fatalities is an integer count >= 0
    df['fatalities'] = df['fatalities'].fillna(0).astype(int)

    # Keep only rows with non-negative fatalities
    df = df[df['fatalities'] >= 0]

    # Return a dataframe with all variables used in modeling plus useful identifiers
    keep_cols = [
        'id', 'name', 'source', 'masfem', 'masfem_z', 'female_name', 'mturk_masfem',
        'max_wind', 'min_pressure', 'category', 'year', 'year_centered',
        'fatalities', 'damage_2015', 'log_damage2015', 'years_since'
    ]
    # Some columns may not exist in alternative versions; select those that do
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit count outcome models to test whether more-feminine hurricane names are associated with more
    fatalities (hypothesized mechanism: feminine names perceived as less threatening -> fewer precautions -> more fatalities).

    Modeling approach:
      1) Fit a Poisson GLM with robust (HC3) standard errors.
      2) Compute Poisson dispersion (Pearson chi-square / df_resid). If there is overdispersion (>~1.5), fit a Negative Binomial GLM.

    Formula:
      fatalities ~ masfem_z + female_name + max_wind + min_pressure + C(category) + year_centered + log_damage2015

    Returns:
      A dict with Poisson results, Negative Binomial results, and dispersion statistic. Each fitted result is a statsmodels results object.
    """
    import statsmodels.api as sm

    # Define formula (we include key controls). If log_damage2015 is present, include it; else omit.
    base_terms = ['masfem_z', 'female_name', 'max_wind', 'min_pressure', 'C(category)', 'year_centered']
    if 'log_damage2015' in df.columns:
        base_terms.append('log_damage2015')
    formula = 'fatalities ~ ' + ' + '.join(base_terms)

    # Fit Poisson with robust SE
    poisson_model = sm.GLM.from_formula(formula, data=df, family=sm.families.Poisson())
    poisson_res = poisson_model.fit(cov_type='HC3')

    # Compute Pearson chi-square dispersion to assess overdispersion
    # Use model-predicted mu from the fitted Poisson
    try:
        mu = poisson_res.mu
    except Exception:
        # If attribute not present, compute predicted mean via model.predict
        mu = poisson_res.predict()
    y = df['fatalities'].values
    # Avoid division by zero in variance calculation: only include mu>0 entries
    valid = mu > 0
    if valid.sum() > 0:
        pearson_chi2 = (((y[valid] - mu[valid]) ** 2) / mu[valid]).sum()
        dispersion = pearson_chi2 / float(poisson_res.df_resid) if poisson_res.df_resid > 0 else np.nan
    else:
        pearson_chi2 = np.nan
        dispersion = np.nan

    # Fit Negative Binomial as a more flexible alternative for overdispersed counts
    nb_res = None
    try:
        nb_model = sm.GLM.from_formula(formula, data=df, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit(cov_type='HC3')
    except Exception as e:
        # If NegativeBinomial fails, leave nb_res as None but report the error
        nb_res = {'error': str(e)}

    # Print short summaries to help interpretation (users can inspect .summary() on returned results)
    print('Poisson model summary:')
    print(poisson_res.summary())
    print('\nDispersion (Pearson chi2 / df_resid): {:.3f}'.format(dispersion if not np.isnan(dispersion) else -1))
    if isinstance(nb_res, dict) and 'error' in nb_res:
        print('\nNegative Binomial fit failed:', nb_res['error'])
    else:
        print('\nNegative Binomial model summary:')
        print(nb_res.summary())

    # Return results for programmatic use
    return {
        'formula': formula,
        'poisson_results': poisson_res,
        'nb_results': nb_res,
        'pearson_chi2': pearson_chi2,
        'dispersion': dispersion
    }


