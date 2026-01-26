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
    """
    Transform the raw Fair (Psychology Today survey) dataframe into the modeling dataframe.

    Produces/ensures the following columns used in modeling:
      - affairs: original count of affairs (kept as-is)
      - any_affair: binary indicator (1 if affairs > 0, else 0)
      - children_binary: 1 if children == 'yes', 0 if 'no'
      - gender_male: 1 if gender == 'male', 0 if 'female'
      - age, yearsmarried, religiousness, education, occupation, rating: kept as numeric controls

    Drops rows with missing values in any of the columns used by the models.
    """
    df = df.copy()

    # Ensure expected columns exist
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Normalize string columns to lower-case for mapping
    if df['children'].dtype == object or pd.api.types.is_categorical_dtype(df['children']):
        df['children_clean'] = df['children'].astype(str).str.strip().str.lower()
    else:
        df['children_clean'] = df['children'].astype(str).str.strip().str.lower()

    if df['gender'].dtype == object or pd.api.types.is_categorical_dtype(df['gender']):
        df['gender_clean'] = df['gender'].astype(str).str.strip().str.lower()
    else:
        df['gender_clean'] = df['gender'].astype(str).str.strip().str.lower()

    # Map children to binary (1=yes, 0=no). If unknown values present, they will produce NaN and later be dropped.
    df['children_binary'] = df['children_clean'].map({'yes': 1, 'y': 1, 'no': 0, 'n': 0})

    # Map gender to binary male indicator (male=1, female=0)
    df['gender_male'] = df['gender_clean'].map({'male': 1, 'm': 1, 'female': 0, 'f': 0})

    # Create binary indicator for any extramarital affair
    # Ensure affairs is numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    df['any_affair'] = (df['affairs'] > 0).astype(int)

    # Ensure numeric control columns are numeric
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any modeling column
    model_cols = ['affairs', 'any_affair', 'children_binary', 'gender_male'] + numeric_cols
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    # Return dataframe with added columns (keep original columns as well)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs two related analyses to answer whether having children decreases engagement in extramarital affairs.

    1) Logistic regression for the probability of any affair (binary outcome any_affair).
       Dependent variable: any_affair
       Independent variable: children_binary
       Controls: gender_male, age, yearsmarried, religiousness, education, occupation, rating

    2) Negative binomial regression for the count/frequency of affairs among respondents who reported any (affairs > 0).
       Dependent variable: affairs (count, positive subset)
       Same independent variables and controls.

    Returns a dict with fitted model objects and a simple average marginal effect estimate for children on probability of any affair.
    """
    results = {}

    # Define covariates used in both models
    covariates = ['children_binary', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # Prepare design matrix for logistic regression (probability of any affair)
    X = df[covariates]
    X = sm.add_constant(X, has_constant='add')
    y = df['any_affair']

    # Fit logistic regression (Logit)
    try:
        logit_model = sm.Logit(y, X).fit(disp=False)
    except Exception:
        # Fallback to using method='lbfgs' if default fails
        logit_model = sm.Logit(y, X).fit(method='lbfgs', disp=False)

    results['logit_model'] = logit_model

    # Compute a simple average marginal effect for children on predicted probability of any affair
    # Evaluate predicted probability at mean values of covariates with children_binary set to 1 vs 0
    X_mean = X.mean().to_frame().T  # single-row DataFrame of means, includes const
    X_mean_c1 = X_mean.copy()
    X_mean_c1['children_binary'] = 1
    X_mean_c0 = X_mean.copy()
    X_mean_c0['children_binary'] = 0
    p1 = logit_model.predict(X_mean_c1)[0]
    p0 = logit_model.predict(X_mean_c0)[0]
    results['marginal_effect_children_on_prob_any_affair'] = (p1 - p0)
    results['predicted_prob_with_children_at_means'] = p1
    results['predicted_prob_without_children_at_means'] = p0

    # Negative binomial for counts among those with > 0 affairs
    pos = df[df['affairs'] > 0].copy()
    if len(pos) >= 10:
        X_pos = pos[covariates]
        X_pos = sm.add_constant(X_pos, has_constant='add')
        y_pos = pos['affairs']

        # Fit GLM Negative Binomial
        try:
            nb_model = sm.GLM(y_pos, X_pos, family=sm.families.NegativeBinomial()).fit()
        except Exception:
            # If NB fails, fall back to Poisson with robust covariances
            nb_model = sm.GLM(y_pos, X_pos, family=sm.families.Poisson()).fit(cov_type='HC0')

        results['nb_model'] = nb_model
    else:
        results['nb_model'] = None
        results['nb_note'] = 'Too few positive-affair observations to fit a stable count model.'

    # Return results dictionary (contains fitted model objects and scalar summaries)
    return results


