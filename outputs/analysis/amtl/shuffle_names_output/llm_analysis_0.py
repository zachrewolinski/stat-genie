from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/shuffle_names_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # --- Map columns (dataset schema appears to have mis-labeled fields). We implement robust fallbacks.
    # Based on schema mapping: 'stdev_age' holds the count of missing teeth (AMTL_count),
    # 'prob_male' holds the number of observable sockets, 'num_amtl' holds estimated age,
    # 'pop' holds sex probability (0-1), 'age' holds specimen genus, and 'genus' holds tooth class.

    # AMTL count
    if 'stdev_age' in df.columns:
        df['AMTL_count'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    elif 'num_amtl' in df.columns and df['num_amtl'].dtype.kind in 'iuf':
        # fallback (if provided differently)
        df['AMTL_count'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    else:
        df['AMTL_count'] = np.nan

    # Sockets (number of observable sockets)
    if 'prob_male' in df.columns:
        df['Sockets'] = pd.to_numeric(df['prob_male'], errors='coerce')
    elif 'sockets' in df.columns:
        df['Sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    else:
        df['Sockets'] = np.nan

    # Age in years (estimated age at death)
    if 'num_amtl' in df.columns:
        df['AgeYears'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    elif 'stdev_age' in df.columns and df['stdev_age'].dtype.kind in 'f':
        df['AgeYears'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    else:
        df['AgeYears'] = np.nan

    # Sex probability / estimate
    if 'pop' in df.columns:
        df['SexProbMale'] = pd.to_numeric(df['pop'], errors='coerce')
    elif 'prob_male' in df.columns and df['prob_male'].between(0,1).any():
        df['SexProbMale'] = pd.to_numeric(df['prob_male'], errors='coerce')
    else:
        df['SexProbMale'] = np.nan

    # Genus (taxon) - according to the schema the column 'age' contains genus names
    if 'age' in df.columns:
        df['Genus'] = df['age'].astype(str)
    elif 'genus' in df.columns and df['genus'].str.contains('Homo|Pan|Pongo|Papio', case=False, na=False).any():
        df['Genus'] = df['genus'].astype(str)
    else:
        df['Genus'] = np.nan

    # Tooth class - according to schema the column 'genus' actually contains tooth class labels
    if 'genus' in df.columns:
        df['ToothClass'] = df['genus'].astype(str)
    elif 'tooth_class' in df.columns:
        df['ToothClass'] = df['tooth_class'].astype(str)
    else:
        df['ToothClass'] = np.nan

    # Population / region (kept for potential later use)
    if 'tooth_class' in df.columns:
        df['Population'] = df['tooth_class'].astype(str)
    elif 'specimen' in df.columns:
        df['Population'] = np.nan
    else:
        df['Population'] = np.nan

    # Clean numeric fields and coerce to sensible types
    # Round AMTL_count and Sockets to integers (counts)
    df['AMTL_count'] = pd.to_numeric(df['AMTL_count'], errors='coerce')
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce')

    # If AMTL_count is fractional because of mapping issues, round to nearest int
    df.loc[df['AMTL_count'].notnull(), 'AMTL_count'] = df.loc[df['AMTL_count'].notnull(), 'AMTL_count'].round().astype('Int64')
    df.loc[df['Sockets'].notnull(), 'Sockets'] = df.loc[df['Sockets'].notnull(), 'Sockets'].round().astype('Int64')

    # Ensure non-negative
    df.loc[df['AMTL_count'] < 0, 'AMTL_count'] = 0
    df.loc[df['Sockets'] < 0, 'Sockets'] = np.nan

    # If both AMTL_count and Sockets are available, enforce AMTL_count <= Sockets
    mask_valid_counts = df['AMTL_count'].notnull() & df['Sockets'].notnull()
    df.loc[mask_valid_counts & (df['AMTL_count'] > df['Sockets']), 'AMTL_count'] = df.loc[mask_valid_counts & (df['AMTL_count'] > df['Sockets']), 'Sockets']

    # Sex binary: derive SexMale from SexProbMale if available
    df['SexProbMale'] = pd.to_numeric(df['SexProbMale'], errors='coerce')
    df['SexMale'] = np.where(df['SexProbMale'].notnull(), (df['SexProbMale'] >= 0.5).astype('Int64'), pd.NA)

    # Create IsHuman indicator
    df['IsHuman'] = df['Genus'].str.contains('Homo', case=False, na=False).astype('Int64')

    # Standardize Genus and ToothClass to categories with consistent capitalization
    df['Genus'] = df['Genus'].replace({'nan': None})
    df['Genus'] = df['Genus'].where(df['Genus'].notnull(), None)
    df['ToothClass'] = df['ToothClass'].replace({'nan': None})

    # Drop rows that cannot be used in the binomial model (need both AMTL_count and Sockets and Genus and ToothClass)
    required = ['AMTL_count', 'Sockets', 'Genus', 'ToothClass']
    df = df.dropna(subset=required)

    # Keep integer types for counts
    df['AMTL_count'] = df['AMTL_count'].astype(int)
    df['Sockets'] = df['Sockets'].astype(int)

    # Remove rows with zero sockets (no trials)
    df = df[df['Sockets'] > 0]

    # Final columns kept for analysis
    keep_cols = ['AMTL_count', 'Sockets', 'Genus', 'IsHuman', 'AgeYears', 'SexProbMale', 'SexMale', 'ToothClass', 'Population', 'specimen']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # df is the transformed dataframe returned by transform()
    # Build binomial response as an (n_obs, 2) array: [successes, failures]
    endog = np.vstack([df['AMTL_count'].values, (df['Sockets'] - df['AMTL_count']).values]).T

    # Construct design matrix: genus (categorical), tooth class (categorical), sex, age
    # Use one-hot encoding for categorical predictors, dropping the first level to avoid multicollinearity
    cat_df = pd.get_dummies(df[['Genus', 'ToothClass']], drop_first=True)

    # Add controls
    X = cat_df.copy()

    # SexMale (binary) - if missing, treat as 0/NA; fill NA with 0.5 (neutral) for model fitting or drop rows earlier
    X['SexMale'] = pd.to_numeric(df['SexMale'].fillna(0), errors='coerce').astype(float)

    # AgeYears - continuous; center and scale for numerical stability if available
    if df['AgeYears'].notnull().any():
        age = pd.to_numeric(df['AgeYears'], errors='coerce')
        age_mean = age.mean()
        age_std = age.std(ddof=0) if age.std(ddof=0) > 0 else 1.0
        X['AgeYears_sc'] = ((age - age_mean) / age_std).fillna(0).astype(float)
    else:
        X['AgeYears_sc'] = 0.0

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit binomial GLM with logit link using successes/failures 2-column endog
    model_glm = sm.GLM(endog, X, family=sm.families.Binomial())
    results = model_glm.fit()

    # For convenience, also compute and attach a simple human vs non-human contrast model
    # Build a small model with IsHuman + controls
    df2 = df.copy()
    X2 = pd.get_dummies(df2['ToothClass'], drop_first=True)
    X2['IsHuman'] = df2['IsHuman'].astype(int)
    X2['SexMale'] = df2['SexMale'].fillna(0).astype(int)
    if df2['AgeYears'].notnull().any():
        age = pd.to_numeric(df2['AgeYears'], errors='coerce')
        X2['AgeYears_sc'] = ((age - age.mean()) / (age.std(ddof=0) if age.std(ddof=0)>0 else 1)).fillna(0)
    else:
        X2['AgeYears_sc'] = 0.0
    X2 = sm.add_constant(X2, has_constant='add')
    model_human = sm.GLM(np.vstack([df2['AMTL_count'].values, (df2['Sockets'] - df2['AMTL_count']).values]).T,
                         X2, family=sm.families.Binomial())
    results_human = model_human.fit()

    # Return both the full genus model and the focused human-vs-nonhuman model
    return {
        'full_genus_model': results,
        'human_contrast_model': results_human
    }


