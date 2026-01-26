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
    """
    Transform the raw Fair affairs dataset into a clean dataframe for modeling.

    Produces these final columns used in models:
      - affairs (original count, numeric)
      - affair_any (binary: 1 if affairs > 0 else 0)
      - children_bin (1 if children == 'yes', 0 if 'no')
      - gender_male (1 if gender == 'male', 0 if 'female')
      - age (numeric)
      - yearsmarried (numeric)
      - religiousness (numeric)
      - education (numeric)
      - occupation (numeric)
      - rating (numeric)

    The function drops rows with missing values in any variable used by either model.
    """
    df = df.copy()

    # Standardize column names if needed (most are already as given)
    # Ensure key columns exist
    required_columns = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input df: {missing}")

    # Clean and coerce types
    # affairs: ensure numeric (already numeric in schema), but coerce and clip to non-negative
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    df.loc[df['affairs'] < 0, 'affairs'] = np.nan

    # Create binary dependent variable: any affair vs none
    df['affair_any'] = (df['affairs'] > 0).astype(int)

    # Independent variable: children -> binary
    # Accept variations in capitalization
    df['children_bin'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Gender binary: male=1, female=0 (if other categories present, they become NaN)
    df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # Coerce other controls to numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['yearsmarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['education'] = pd.to_numeric(df['education'], errors='coerce')
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows with missing values in any of the columns used in models
    model_cols = [
        'affairs', 'affair_any', 'children_bin', 'gender_male', 'age',
        'yearsmarried', 'religiousness', 'education', 'occupation', 'rating'
    ]
    df = df.dropna(subset=model_cols)

    # Optional: cast to appropriate dtypes
    df['children_bin'] = df['children_bin'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)
    df['affair_any'] = df['affair_any'].astype(int)

    # Return the transformed dataframe with all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs a two-part analysis to answer: Does having children decrease engagement in extramarital affairs?

    1) Main analysis: logistic regression on affair_any (any affair vs none) with children_bin as primary IV
    2) Secondary analysis: among respondents who reported at least one affair, model the count of affairs using a negative binomial regression (to account for overdispersion).

    Returns a dictionary with fitted result objects and brief summary numbers.
    """
    results = {}

    # Ensure df is the transformed df with required columns
    required = ['affair_any', 'affairs', 'children_bin', 'gender_male', 'age',
                'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Explanatory variables / controls
    controls = ['gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    exog_vars = ['children_bin'] + controls

    # Build design matrix with constant
    X = df[exog_vars]
    X = sm.add_constant(X, has_constant='add')

    # 1) Logistic regression: any affair
    y_logit = df['affair_any']
    logit_model = sm.Logit(y_logit, X)
    logit_res = logit_model.fit(disp=False)

    # Compute odds ratio and 95% CI for children effect
    coef = logit_res.params['children_bin']
    se = logit_res.bse['children_bin']
    or_children = np.exp(coef)
    ci_lower = np.exp(coef - 1.96 * se)
    ci_upper = np.exp(coef + 1.96 * se)

    results['logit_result'] = logit_res
    results['logit_OR_children'] = {
        'odds_ratio': float(or_children),
        '95%_CI_lower': float(ci_lower),
        '95%_CI_upper': float(ci_upper),
        'pvalue': float(logit_res.pvalues['children_bin'])
    }

    # 2) Count model among those with at least one affair: Negative Binomial
    df_pos = df[df['affairs'] > 0].copy()
    if df_pos.shape[0] >= 30:
        X_pos = sm.add_constant(df_pos[exog_vars], has_constant='add')
        y_pos = df_pos['affairs']
        # Use statsmodels discrete NegativeBinomial
        try:
            nb_model = sm.NegativeBinomial(y_pos, X_pos)
            nb_res = nb_model.fit(disp=False)
            results['nb_result'] = nb_res
            # exponentiated coefficient (incidence rate ratio) for children
            coef_nb = nb_res.params['children_bin']
            irr = np.exp(coef_nb)
            se_nb = nb_res.bse['children_bin']
            irr_ci_lower = np.exp(coef_nb - 1.96 * se_nb)
            irr_ci_upper = np.exp(coef_nb + 1.96 * se_nb)
            results['nb_children_IRR'] = {
                'IRR': float(irr),
                '95%_CI_lower': float(irr_ci_lower),
                '95%_CI_upper': float(irr_ci_upper),
                'pvalue': float(nb_res.pvalues['children_bin'])
            }
        except Exception as e:
            # If NegativeBinomial fails to converge or is unavailable, fall back to Poisson with robust SEs
            poi_model = sm.GLM(y_pos, X_pos, family=sm.families.Poisson())
            poi_res = poi_model.fit(cov_type='HC0', disp=False)
            results['nb_result'] = poi_res
            coef_p = poi_res.params['children_bin']
            irr_p = np.exp(coef_p)
            se_p = poi_res.bse['children_bin']
            results['nb_children_IRR'] = {
                'IRR': float(irr_p),
                '95%_CI_lower': float(np.exp(coef_p - 1.96 * se_p)),
                '95%_CI_upper': float(np.exp(coef_p + 1.96 * se_p)),
                'pvalue': float(poi_res.pvalues['children_bin'])
            }
    else:
        results['nb_result'] = None
        results['nb_children_IRR'] = None
        results['nb_warning'] = 'Too few positive-affair observations to fit a reliable count model (need >=30 suggested).'

    # Include sample sizes
    results['n_total'] = int(df.shape[0])
    results['n_any_affair'] = int(df['affair_any'].sum())
    results['n_children_yes'] = int(df['children_bin'].sum())

    return results


