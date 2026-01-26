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
    Transform the raw Fair affairs dataset into a dataframe ready for modeling.

    Produces the following new columns used by the model:
      - Children: binary 1 if 'children' == 'yes', 0 if 'no'
      - gender_female: binary 1 if 'gender' == 'female', 0 if 'male'
      - AnyAffair: binary 1 if affairs > 0, 0 otherwise (used for robustness logistic model)

    Keeps original 'affairs' as the dependent count variable.
    Drops rows missing any modeling variable.
    """
    df = df.copy()

    # Ensure affairs numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children to binary indicator
    # Accept common variants; if unexpected values appear, map to NaN so we drop them
    df['Children'] = df['children'].map({
        'yes': 1,
        'no': 0
    })

    # Map gender to binary female indicator
    df['gender_female'] = df['gender'].map({
        'female': 1,
        'male': 0
    })

    # Derived binary outcome for robustness check (any affair or none)
    df['AnyAffair'] = (df['affairs'] > 0).astype(int)

    # Select variables required for the models
    required_cols = [
        'affairs',
        'Children',
        'gender_female',
        'age',
        'yearsmarried',
        'religiousness',
        'education',
        'occupation',
        'rating',
        'AnyAffair'
    ]

    # Coerce numeric control columns to numeric where appropriate
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing required modeling variables
    df = df.dropna(subset=required_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to estimate relationship between having children and extramarital affairs.

    Models fitted:
      1) Negative binomial regression (GLM) for count outcome 'affairs' (primary model for overdispersed counts).
      2) Logistic regression for binary outcome 'AnyAffair' (robustness check: probability of any affair).
      3) Zero-inflated negative binomial (optional robustness) to account for excess zeros if supported.

    Returns a dictionary with fitted model results objects (statsmodels results). Caller can inspect .summary().
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.tools import add_constant

    results = {}

    # Ensure the dataframe contains the columns we expect
    needed = ['affairs', 'Children', 'gender_female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'AnyAffair']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Define covariates (controls)
    covariate_cols = ['Children', 'gender_female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    X = add_constant(df[covariate_cols], has_constant='add')

    # Dependent variables
    y_count = df['affairs'].astype(float)
    y_bin = df['AnyAffair'].astype(int)

    # 1) Negative binomial GLM (log link) for counts
    try:
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        results['neg_binom'] = nb_res
    except Exception as e:
        results['neg_binom_error'] = f'NegativeBinomial model failed: {e}'

    # 2) Logistic regression for any affair (robustness)
    try:
        logit_model = sm.Logit(y_bin, X)
        logit_res = logit_model.fit(disp=False)
        results['logit_any_affair'] = logit_res
    except Exception as e:
        # try GLM binomial if Logit fails to converge
        try:
            glm_binom = sm.GLM(y_bin, X, family=sm.families.Binomial()).fit()
            results['logit_any_affair'] = glm_binom
        except Exception as e2:
            results['logit_any_affair_error'] = f'Logit/GLM-Binomial failed: {e}; {e2}'

    # 3) Zero-inflated negative binomial (optional, for excess zeros). Use statsmodels' count_model if available.
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
        # Use same regressors for count and inflation parts (could customize)
        zinb = ZeroInflatedNegativeBinomialP(endog=y_count, exog=X, exog_infl=X, inflation='logit')
        zinb_res = zinb.fit(disp=False)
        results['zinb'] = zinb_res
    except Exception as e:
        results['zinb_error'] = f'ZINB not available or failed: {e}'

    # Print brief summaries for quick inspection (caller can use returned objects to get full summaries)
    print('\n--- Negative Binomial summary (if available) ---')
    if 'neg_binom' in results:
        print(results['neg_binom'].summary())
    else:
        print(results.get('neg_binom_error'))

    print('\n--- Logistic / Binomial summary (if available) ---')
    if 'logit_any_affair' in results:
        print(results['logit_any_affair'].summary())
    else:
        print(results.get('logit_any_affair_error'))

    print('\n--- Zero-Inflated Negative Binomial summary (if available) ---')
    if 'zinb' in results:
        print(results['zinb'].summary())
    else:
        print(results.get('zinb_error'))

    return results


