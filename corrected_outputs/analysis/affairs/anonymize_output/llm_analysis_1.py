from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe containing the variables needed for analysis.

    Input columns expected in raw df (per schema):
      - feature2: frequency of extramarital intercourse in past year (0,1,2,3,7,12,...)
      - feature3: gender (categorical: 'female'/'male')
      - feature4: age code (numeric)
      - feature5: years married (numeric-coded categories)
      - feature6: children in marriage (categorical: 'yes'/'no')
      - feature7: religiousness (1-5)
      - feature8: education (numeric code)
      - feature9: occupation (numeric code)
      - feature10: marriage happiness (1-5)

    Returns a dataframe with columns used in modeling:
      - Affairs (int count), LogAffairs (float), Children (0/1), Female (0/1),
        Children_Female (interaction term), Age, YearsMarried, Religiousness,
        Education, Occupation, MarriageHappiness
    """
    # Work on a copy
    df = df.copy()

    # Ensure key columns exist
    required = ['feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Convert numeric columns to numeric (coerce errors to NaN)
    df['Affairs'] = pd.to_numeric(df['feature2'], errors='coerce')
    df['Age'] = pd.to_numeric(df['feature4'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Education'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['MarriageHappiness'] = pd.to_numeric(df['feature10'], errors='coerce')

    # Create binary Children indicator from feature6; handle capitalization and possible variants
    df['Children'] = df['feature6'].apply(lambda x: 1 if (isinstance(x, str) and x.strip().lower() in ['yes','y','1','true']) else (0 if (isinstance(x, str) and x.strip().lower() in ['no','n','0','false']) else np.nan))

    # Create Female indicator from feature3
    df['Female'] = df['feature3'].apply(lambda x: 1 if (isinstance(x, str) and x.strip().lower() == 'female') else (0 if (isinstance(x, str) and x.strip().lower() == 'male') else np.nan))

    # Drop rows with missing values in the variables needed for the main model
    cols_needed = ['Affairs','Children','Female','Age','YearsMarried','Religiousness','Education','Occupation','MarriageHappiness']
    df = df.dropna(subset=cols_needed)

    # Ensure Affairs is integer and non-negative
    df['Affairs'] = df['Affairs'].astype(int)
    df.loc[df['Affairs'] < 0, 'Affairs'] = 0

    # Create log-transformed dependent for OLS robustness (log1p preserves zeros)
    df['LogAffairs'] = np.log1p(df['Affairs'].astype(float))

    # Interaction term: Children x Female (to test moderation by gender)
    df['Children_Female'] = df['Children'] * df['Female']

    # Reorder/keep only final columns for modeling
    final_cols = ['Affairs','LogAffairs','Children','Female','Children_Female','Age','YearsMarried','Religiousness','Education','Occupation','MarriageHappiness']
    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to estimate the relationship between having children and extramarital affairs.

    Primary model: Zero-Inflated Negative Binomial (accounts for many zeros and overdispersion in count outcome).
    Robustness model: OLS on log1p(Affairs) with same covariates.

    Returns a dictionary with fitted results objects: {'zinb_res': <results>, 'ols_res': <results>}.
    """
    # Required columns check
    required = ['Affairs','LogAffairs','Children','Female','Children_Female','Age','YearsMarried','Religiousness','Education','Occupation','MarriageHappiness']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare exogenous matrices
    exog_vars = ['Children','Female','Children_Female','Age','YearsMarried','Religiousness','Education','Occupation','MarriageHappiness']
    X = df[exog_vars].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # For the inflation (zero) model, include a subset of predictors (no interaction to aid convergence)
    exog_infl_vars = ['Children','Female','Age','YearsMarried','Religiousness','Education','Occupation','MarriageHappiness']
    X_infl = df[exog_infl_vars].astype(float)
    X_infl = sm.add_constant(X_infl, has_constant='add')

    endog = df['Affairs'].astype(int)

    results = {}

    # 1) Zero-Inflated Negative Binomial (primary)
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
        zinb_mod = ZeroInflatedNegativeBinomialP(endog, X, exog_infl=X_infl, p=1)
        zinb_res = zinb_mod.fit(method='newton', maxiter=100, disp=0)
        results['zinb_res'] = zinb_res
    except Exception as e:
        # If ZINB is unavailable or fails to converge, fall back to Zero-Inflated Poisson
        try:
            from statsmodels.discrete.count_model import ZeroInflatedPoisson
            zip_mod = ZeroInflatedPoisson(endog, X, exog_infl=X_infl)
            zip_res = zip_mod.fit(method='newton', maxiter=100, disp=0)
            results['zip_res_fallback'] = zip_res
        except Exception as e2:
            results['inflation_model_error'] = f"Both ZINB and ZIP failed: ZINB error: {e}; ZIP error: {e2}"

    # 2) OLS on log1p(Affairs) as a robustness check
    X_ols = df[['Children','Female','Children_Female','Age','YearsMarried','Religiousness','Education','Occupation','MarriageHappiness']].astype(float)
    X_ols = sm.add_constant(X_ols, has_constant='add')
    ols_mod = sm.OLS(df['LogAffairs'].astype(float), X_ols)
    ols_res = ols_mod.fit()
    results['ols_res'] = ols_res

    return results


