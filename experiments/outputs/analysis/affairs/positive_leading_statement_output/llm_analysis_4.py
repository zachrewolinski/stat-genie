from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/positive_leading_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Fair affairs dataset for count-regression modeling.

    Final dataframe columns (and exact names used in the model):
      - affairs: dependent count variable (int)
      - Children: binary 0/1 (1 = yes)
      - Gender_Male: binary 0/1 (1 = male)
      - Age, YearsMarried, Religiousness, Education, Occupation, Rating: numeric controls
    """
    df = df.copy()

    # Columns required for analysis
    required = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]

    # Keep only relevant columns (if some are missing this will raise KeyError)
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    df = df[required]

    # Drop rows with missing values in the core variables
    df = df.dropna(subset=required)

    # Map binary variables to 0/1
    df['Children'] = df['children'].map({
        'yes': 1,
        'no': 0,
        'Yes': 1,
        'No': 0,
        1: 1,
        0: 0
    })

    df['Gender_Male'] = df['gender'].map({
        'male': 1,
        'female': 0,
        'Male': 1,
        'Female': 0
    })

    # Convert numeric columns to numeric dtype (coerce bad values to NaN)
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop any rows where mapping or numeric coercion failed
    df = df.dropna(subset=['Children', 'Gender_Male'] + numeric_cols)

    # Cast affairs to integer (it is a count variable). Keep as int for modeling.
    df['affairs'] = df['affairs'].astype(int)

    # Rename some columns for consistent naming in model
    df = df.rename(columns={
        'age': 'Age',
        'yearsmarried': 'YearsMarried',
        'religiousness': 'Religiousness',
        'education': 'Education',
        'occupation': 'Occupation',
        'rating': 'Rating'
    })

    # Keep and return only the columns needed for modeling, in a fixed order
    final_cols = [
        'affairs', 'Children', 'Gender_Male', 'Age', 'YearsMarried',
        'Religiousness', 'Education', 'Occupation', 'Rating'
    ]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count regression models to estimate the association between having children and number
    of extramarital affairs, controlling for covariates and testing for gender moderation.

    Procedure:
      1. Fit Poisson GLM (log link).
      2. Compute Pearson-based dispersion statistic to assess overdispersion.
      3. Fit Negative Binomial GLM if overdispersion is present (and report both models).

    Returns a dictionary containing fitted model objects and diagnostics.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    # Ensure required columns are present
    required = ['affairs', 'Children', 'Gender_Male', 'Age', 'YearsMarried',
                'Religiousness', 'Education', 'Occupation', 'Rating']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Transformed dataframe is missing required columns: {missing}")

    # Formula includes interaction to test whether effect of Children differs by gender
    formula = 'affairs ~ Children * Gender_Male + Age + YearsMarried + Religiousness + Education + Occupation + Rating'

    # Fit Poisson model
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()

    # Compute Pearson chi-square dispersion statistic for Poisson
    mu = poisson_model.predict(df)
    y = df['affairs']
    # Avoid division by zero for any predicted mu that are zero
    eps = 1e-8
    pearson_chi2 = (((y - mu) ** 2) / (mu + eps)).sum()
    dispersion = pearson_chi2 / poisson_model.df_resid

    # Fit Negative Binomial model (addresses overdispersion). Always fit for comparison.
    # statsmodels' NegativeBinomial family estimates a variance function with an extra parameter.
    nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()

    # Prepare a concise summary dictionary
    results = {
        'poisson_model': poisson_model,
        'nb_model': nb_model,
        'poisson_dispersion_pearson_chi2_div_df': dispersion,
        'formula': formula,
        'n_obs': int(df.shape[0])
    }

    # Print short diagnostics for users running this function interactively
    print('Observations:', results['n_obs'])
    print('Poisson dispersion (Pearson chi2 / df):', round(dispersion, 3))
    print('\n--- Poisson model summary ---')
    print(poisson_model.summary())
    print('\n--- Negative Binomial model summary ---')
    print(nb_model.summary())

    return results


