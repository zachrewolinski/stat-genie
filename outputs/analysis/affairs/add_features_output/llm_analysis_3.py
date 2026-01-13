from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep only columns needed for analysis and drop rows with missing values in these columns
    needed = [
        'affairs',        # original outcome
        'children',       # original IV (yes/no)
        'gender',         # original gender factor
        'age',
        'yearsmarried',
        'religiousness',
        'education',
        'occupation',
        'rating'
    ]
    # Some datasets may contain these columns but with NaNs; drop incomplete rows for these fields
    df = df.loc[:, df.columns.isin(needed)].copy()
    df = df.dropna(subset=needed)

    # Dependent variable: ensure numeric count of affairs
    df['affairs_count'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Independent variable: children -> binary indicator (1 = yes, 0 = no)
    # Accept common lowercase/uppercase strings; map unknowns to NaN
    df['Children'] = df['children'].map(lambda x: 1 if str(x).strip().lower() == 'yes' else (0 if str(x).strip().lower() == 'no' else np.nan))

    # Control: gender -> binary male=1 female=0 (map robustly)
    df['GenderMale'] = df['gender'].map(lambda x: 1 if str(x).strip().lower() in ['male','m'] else (0 if str(x).strip().lower() in ['female','f'] else np.nan))

    # Numeric controls: coerce to numeric and keep as-is
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows that became NA after coercion (we need complete cases for the chosen model)
    df = df.dropna(subset=['affairs_count', 'Children', 'GenderMale', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating'])

    # Cast integer-like columns to integers where appropriate
    df['affairs_count'] = df['affairs_count'].astype(int)
    df['Children'] = df['Children'].astype(int)
    df['GenderMale'] = df['GenderMale'].astype(int)

    # Final dataframe returned contains the new/clean columns used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Fit a Negative Binomial regression for count outcome with robust comparison to Poisson
    # This models the (overdispersed) count of affairs as a function of having children and controls.
    import statsmodels.api as sm

    # Work with a copy
    df = df.copy()

    # Outcome
    y = df['affairs_count']

    # Predictors: main IV + controls (constant added)
    X = df[[
        'Children',
        'GenderMale',
        'Age',
        'YearsMarried',
        'Religiousness',
        'Education',
        'Occupation',
        'Rating'
    ]]
    X = sm.add_constant(X)

    # Negative Binomial (preferred for overdispersed counts)
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()

    # Poisson (for comparison / robustness) -- often underestimates variance if overdispersion is present
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    # Return both fitted results objects for inspection (summary, params, pvalues, etc.)
    return {
        'negative_binomial': nb_model,
        'poisson': poisson_model
    }


