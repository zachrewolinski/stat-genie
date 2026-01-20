from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep only rows with required fields present
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
    df = df.dropna(subset=required)

    # Ensure sockets is positive integer > 0
    df = df[df['sockets'] > 0]

    # Ensure num_amtl is non-negative and does not exceed sockets
    # If any num_amtl > sockets, cap to sockets (more conservative than drop)
    df['num_amtl'] = df['num_amtl'].astype(float)
    df['sockets'] = df['sockets'].astype(float)
    df['num_amtl'] = np.minimum(df['num_amtl'], df['sockets'])
    df['num_amtl'] = np.maximum(df['num_amtl'], 0.0)

    # Proportion missing within the tooth class (dependent variable)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Create binary indicator variable for modern humans (Homo sapiens)
    # Strip whitespace and compare case-insensitively
    df['genus'] = df['genus'].astype(str).str.strip()
    df['IsHomo'] = (df['genus'].str.lower() == 'homo sapiens').astype(int)

    # Center age to improve interpretability and reduce collinearity
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure prob_male is numeric and in [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df.loc[df['prob_male'] < 0, 'prob_male'] = 0.0
    df.loc[df['prob_male'] > 1, 'prob_male'] = 1.0

    # Make tooth_class a categorical variable with consistent categories
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().astype('category')

    # Keep only rows where proportion is defined (safeguard)
    df = df.dropna(subset=['prop_amtl', 'IsHomo', 'age_c', 'prob_male', 'tooth_class'])

    # Return transformed dataframe containing all columns needed for the model
    # (we keep original columns too for traceability)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM for AMTL frequency.

    Model specification:
      prop_amtl ~ IsHomo + age_c + prob_male + C(tooth_class)

    The binomial family is used with the number of trials given by 'sockets'.
    The coefficient on IsHomo tests whether modern humans have higher AMTL rates
    controlling for age, sex (prob_male), and tooth class.
    """
    df = df.copy()

    # Build and fit GLM: model proportion with Binomial family and use sockets as weights
    # (weights indicate number of trials for the proportion)
    formula = 'prop_amtl ~ IsHomo + age_c + prob_male + C(tooth_class)'
    model = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial(), weights=df['sockets'])

    # Fit the model; use robust covariance (HC3) for heteroskedasticity-robust SEs
    results = model.fit(cov_type='HC3')

    # Return the fitted results object (user can inspect summary, params, conf_int, etc.)
    return results


