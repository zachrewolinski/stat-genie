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
    # Make a working copy
    df = df.copy()

    # Keep only the columns we need and drop rows with missing critical values
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Standardize / normalize categorical mappings
    # Map children to binary Children: 1 if 'yes', 0 if 'no'
    df['Children'] = df['children'].map({
        'yes': 1,
        'no': 0
    })
    # If mapping yields NaN (unexpected categories), drop those rows
    df = df.dropna(subset=['Children'])

    # Map gender to a binary indicator Gender_Male (1 = male, 0 = female)
    # Accept lowercase/uppercase values defensively
    df['gender_str'] = df['gender'].astype(str).str.lower()
    df['Gender_Male'] = df['gender_str'].map({'male': 1, 'female': 0})
    df = df.dropna(subset=['Gender_Male'])
    df = df.drop(columns=['gender_str'])

    # Create the primary dependent variables
    df['AnyAffair'] = (df['affairs'].astype(float) > 0).astype(int)

    # Center continuous covariates for better interpretability
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()
    df['yearsmarried_c'] = df['yearsmarried'].astype(float) - df['yearsmarried'].astype(float).mean()
    df['religiousness_c'] = df['religiousness'].astype(float) - df['religiousness'].astype(float).mean()

    # Ensure education, occupation, rating are numeric
    df['education'] = pd.to_numeric(df['education'], errors='coerce')
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['education', 'occupation', 'rating'])

    # Interaction term between children and gender
    df['Children_x_Gender'] = df['Children'] * df['Gender_Male']

    # Keep only the final columns used in modeling
    final_cols = ['affairs', 'AnyAffair', 'Children', 'Gender_Male', 'Children_x_Gender',
                  'age_c', 'yearsmarried_c', 'religiousness_c', 'education', 'occupation', 'rating']
    df = df[final_cols]

    # Ensure numeric dtypes
    for c in final_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Final drop of any rows with NaN introduced by coercion
    df = df.dropna()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two complementary models to answer whether having children decreases engagement in extramarital affairs.
    1) Logistic regression for probability of any affair (primary inference).
    2) Negative binomial regression for count of affairs among respondents who report at least one affair (intensity analysis).

    Returns a dict with fitted model result objects and brief summaries.
    """
    results = {}

    # Define covariates and formula columns (must match transform output)
    covariates = ['Children', 'Gender_Male', 'Children_x_Gender', 'age_c', 'yearsmarried_c',
                  'religiousness_c', 'education', 'occupation', 'rating']

    # Prepare design matrix for the full sample (logistic)
    X = df[covariates]
    X = sm.add_constant(X)
    y = df['AnyAffair']

    # Fit logistic regression (binary outcome)
    try:
        logit_mod = sm.Logit(y, X)
        logit_res = logit_mod.fit(disp=False)
        results['logit'] = logit_res
        # Add average marginal effect for Children (and interaction) for interpretability
        try:
            marg = logit_res.get_margeff(at='overall', method='dydx')
            results['logit_margeff'] = marg.summary()
        except Exception:
            results['logit_margeff'] = 'marginal effects could not be computed'
    except Exception as e:
        results['logit'] = None
        results['logit_error'] = str(e)

    # Fit a count model (Negative Binomial) among those with at least one affair to study intensity
    df_pos = df[df['AnyAffair'] == 1].copy()
    if len(df_pos) >= 10:
        Xp = df_pos[covariates]
        Xp = sm.add_constant(Xp)
        yp = df_pos['affairs']
        try:
            nb_mod = sm.GLM(yp, Xp, family=sm.families.NegativeBinomial())
            nb_res = nb_mod.fit()
            results['neg_bin'] = nb_res
        except Exception as e:
            results['neg_bin'] = None
            results['neg_bin_error'] = str(e)
    else:
        results['neg_bin'] = None
        results['neg_bin_note'] = 'Too few positive-affair observations to fit reliable count model.'

    # Summary statistics: raw difference in proportions (simple descriptive check)
    try:
        prop_with_children = df.groupby('Children')['AnyAffair'].mean()
        counts = df.groupby('Children')['AnyAffair'].count()
        results['descriptives'] = pd.DataFrame({'prop_any_affair': prop_with_children, 'n': counts})
    except Exception:
        results['descriptives'] = None

    return results


