from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/noperturb_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) affairs dataset into analysis-ready dataframe.

    Output columns used in modeling:
      - affairs: numeric dependent variable (keeps original coding including top-coded values)
      - children_binary: 1 if 'children' == 'yes', 0 if 'no'
      - gender_male: 1 if male, 0 if female
      - age, yearsmarried, religiousness, education, occupation, rating: numeric controls
      - affair_any: binary indicator (1 if affairs > 0)
    """
    df = df.copy()

    # Standardize column names (in case of leading/trailing spaces or different casing)
    # (Assumes original columns exist as described in schema)

    # Map children to binary
    if 'children' in df.columns:
        df['children_binary'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    else:
        df['children_binary'] = np.nan

    # Map gender to binary male indicator
    if 'gender' in df.columns:
        df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    else:
        df['gender_male'] = np.nan

    # Ensure numeric controls exist and coerce to numeric
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'affairs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Construct a binary indicator for any affair (useful for logistic checks)
    df['affair_any'] = (df['affairs'] > 0).astype(float)

    # Drop rows with missing outcome or main IV or essential controls
    required_for_model = ['affairs', 'children_binary', 'gender_male', 'age', 'yearsmarried',
                          'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_for_model)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to assess whether having children is associated with lower engagement in extramarital affairs:
      1) Zero-Inflated Negative Binomial (ZINB) on the count outcome 'affairs' to account for overdispersion and excess zeros.
      2) Logistic regression on 'affair_any' (any vs none) as a robustness / interpretability check.

    Returns a dictionary with fitted model objects and summary strings.
    """
    # Required imports for modeling
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure transform runs if needed (user may pass raw df)
    # We assume df already transformed by transform(); otherwise user can call transform before model

    # Design matrix for count model
    exog_vars = ['children_binary', 'gender_male', 'age', 'yearsmarried',
                 'religiousness', 'education', 'occupation', 'rating']
    exog = df[exog_vars].astype(float)
    exog = sm.add_constant(exog, has_constant='add')
    endog = df['affairs'].astype(int)

    results = {}

    # 1) Zero-Inflated Negative Binomial
    try:
        zinb_mod = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog, inflation='logit')
        zinb_res = zinb_mod.fit(disp=False, maxiter=100)
        results['zinb_model'] = zinb_res
        results['zinb_summary'] = zinb_res.summary().as_text()
    except Exception as e:
        results['zinb_error'] = str(e)

    # 2) Logistic regression on any affair (binary)
    try:
        formula = 'affair_any ~ ' + ' + '.join(exog_vars)
        logit_res = smf.logit(formula, data=df).fit(disp=False)
        results['logit_model'] = logit_res
        results['logit_summary'] = logit_res.summary().as_text()
    except Exception as e:
        results['logit_error'] = str(e)

    # 3) (Optional) Return descriptive statistics for transparency
    desc = df[['affairs', 'affair_any', 'children_binary'] + exog_vars[1:]].describe().to_dict()
    results['descriptives'] = desc

    return results


