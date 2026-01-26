from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw dataset to the modeling dataframe. Produces the following columns used in models:
      - AnyAffair: binary indicator (1 if feature2 > 0, else 0)
      - NumAffairs: numeric original frequency variable (feature2)
      - Children: 1 if feature6 == 'yes', 0 if 'no'
      - Female: 1 if feature3 == 'female', 0 if 'male'
      - Age: numeric (feature4)
      - YearsMarried: numeric (feature5)
      - Religiosity: numeric (feature7)
      - Education: numeric (feature8)
      - Occupation: numeric (feature9)
      - MaritalHappiness: numeric (feature10)

    Drops rows with missing values in any of the columns above.
    """
    df = df.copy()

    # Create numeric count/frequency of extramarital intercourse
    df['NumAffairs'] = pd.to_numeric(df['feature2'], errors='coerce')

    # Binary indicator: any affair (primary DV)
    df['AnyAffair'] = (df['NumAffairs'] > 0).astype(int)

    # Children present in the marriage (feature6: 'yes'/'no') -> 1/0
    df['Children'] = df['feature6'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Gender: map female/male to 1/0
    df['Female'] = df['feature3'].astype(str).str.lower().map({'female': 1, 'male': 0})

    # Controls: keep numeric versions (coerce errors -> NaN)
    df['Age'] = pd.to_numeric(df['feature4'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['Religiosity'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Education'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['MaritalHappiness'] = pd.to_numeric(df['feature10'], errors='coerce')

    # Select relevant columns for modeling
    model_cols = ['AnyAffair', 'NumAffairs', 'Children', 'Female',
                  'Age', 'YearsMarried', 'Religiosity', 'Education',
                  'Occupation', 'MaritalHappiness']

    # Drop rows with missing values in any modeling column
    df_model = df[model_cols].dropna().reset_index(drop=True)

    # Optional: ensure integer dtypes for binary indicators
    df_model['AnyAffair'] = df_model['AnyAffair'].astype(int)
    df_model['Children'] = df_model['Children'].astype(int)
    df_model['Female'] = df_model['Female'].astype(int)

    return df_model


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits a primary logistic regression for probability of any extramarital affair (AnyAffair)
    and conditional count models (Poisson and Negative Binomial) for NumAffairs among
    respondents who report > 0 affairs.

    Returns a dictionary with fitted results objects and textual summaries:
      - 'logit_res': fitted Logit results (statsmodels)
      - 'logit_summary': text summary
      - 'poisson_res' & 'poisson_summary' (only if there are positive counts)
      - 'negbin_res' & 'negbin_summary' (only if there are positive counts)
    """
    # Expect df is already transformed (output of transform). If not, try to transform.
    required_cols = ['AnyAffair', 'NumAffairs', 'Children', 'Female',
                     'Age', 'YearsMarried', 'Religiosity', 'Education',
                     'Occupation', 'MaritalHappiness']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Build predictors and include interaction Children x Female to test moderation by gender
    X = df[['Children', 'Female', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']].copy()
    X['Children_Female'] = X['Children'] * X['Female']
    X = sm.add_constant(X, has_constant='add')

    y = df['AnyAffair']

    results = {}

    # Logistic regression for any affair
    try:
        logit_model = sm.Logit(y, X)
        logit_res = logit_model.fit(disp=False, maxiter=200)
        results['logit_res'] = logit_res
        results['logit_summary'] = logit_res.summary().as_text()
    except Exception as e:
        results['logit_res'] = None
        results['logit_error'] = str(e)

    # Conditional count models among respondents who reported any affairs (NumAffairs > 0)
    df_pos = df[df['NumAffairs'] > 0].copy()
    if df_pos.shape[0] >= 10:  # require a modest sample size to attempt count models
        X_pos = df_pos[['Children', 'Female', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']].copy()
        X_pos['Children_Female'] = X_pos['Children'] * X_pos['Female']
        X_pos = sm.add_constant(X_pos, has_constant='add')
        y_pos = df_pos['NumAffairs']

        # Poisson
        try:
            poisson_model = sm.GLM(y_pos, X_pos, family=sm.families.Poisson())
            poisson_res = poisson_model.fit()
            results['poisson_res'] = poisson_res
            results['poisson_summary'] = poisson_res.summary().as_text()
        except Exception as e:
            results['poisson_res'] = None
            results['poisson_error'] = str(e)

        # Negative binomial (to allow overdispersion)
        try:
            negbin_model = sm.GLM(y_pos, X_pos, family=sm.families.NegativeBinomial())
            negbin_res = negbin_model.fit()
            results['negbin_res'] = negbin_res
            results['negbin_summary'] = negbin_res.summary().as_text()
        except Exception as e:
            results['negbin_res'] = None
            results['negbin_error'] = str(e)
    else:
        results['poisson_res'] = None
        results['negbin_res'] = None
        results['count_model_note'] = 'Not enough positive-count observations to fit count models (NumAffairs > 0).'

    return results


