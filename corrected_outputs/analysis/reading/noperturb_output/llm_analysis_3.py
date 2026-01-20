from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/noperturb_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataframe into the analysis-ready dataframe.

    Steps performed:
    - Drop rows missing core variables (speed, reader_view, dyslexia_bin, uuid)
    - Ensure binary fields are integer typed
    - Create LogSpeed = log(speed + 1) as the dependent variable
    - Create english_native_bin from 'english_native' (Y->1, N->0)
    - Center continuous controls: num_words, Flesch_Kincaid, age
    - Ensure categorical fields have appropriate dtypes
    - Remove extreme outliers on LogSpeed (>|z|>=4)

    Returns a copy of the dataframe containing the columns named in the conceptual model.
    """
    df = df.copy()

    # Drop rows missing required variables
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin', 'uuid'])

    # Ensure core binary variables are integers
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Dependent variable: log-transform speed to reduce skew
    df['LogSpeed'] = np.log(df['speed'].astype(float) + 1.0)

    # Binary indicator for native English speakers: map 'Y'->1, 'N'->0. Missing -> 0 (non-native) if present.
    df['english_native_bin'] = df.get('english_native').map({'Y': 1, 'N': 0})
    # If english_native column absent or had nan, fillna with 0 and convert to int
    df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)

    # Center continuous controls for interpretability
    # Keep original columns too if needed, but model will use centered versions
    df['num_words_c'] = df['num_words'].astype(float) - df['num_words'].astype(float).mean()
    df['Flesch_Kincaid_c'] = df['Flesch_Kincaid'].astype(float) - df['Flesch_Kincaid'].astype(float).mean()
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()

    # Ensure retake_trial is integer 0/1 and fill missing with 0
    if 'retake_trial' in df.columns:
        df['retake_trial'] = df['retake_trial'].fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Ensure categorical variables have category dtype for modeling
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        df['device'] = 'unknown'
        df['device'] = df['device'].astype('category')

    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype('category')
    else:
        df['page_id'] = 'page_unknown'
        df['page_id'] = df['page_id'].astype('category')

    # Participant id as category (for mixed-effects grouping)
    df['uuid'] = df['uuid'].astype('category')

    # Remove extreme outliers in the transformed DV (e.g., abs(z) >= 4)
    df['LogSpeed_z'] = (df['LogSpeed'] - df['LogSpeed'].mean()) / df['LogSpeed'].std()
    df = df[df['LogSpeed_z'].abs() < 4].copy()
    df.drop(columns=['LogSpeed_z'], inplace=True)

    # Final: ensure all model columns exist
    required_cols = ['LogSpeed', 'reader_view', 'dyslexia_bin', 'num_words_c', 'Flesch_Kincaid_c',
                     'age_c', 'english_native_bin', 'retake_trial', 'device', 'page_id', 'uuid']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns after transform: {missing}")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects model to estimate the effect of Reader View on reading speed and whether
    that effect differs for readers with dyslexia.

    Model specification (primary):
      LogSpeed ~ reader_view * dyslexia_bin + num_words_c + Flesch_Kincaid_c + age_c
                 + english_native_bin + retake_trial + C(device) + C(page_id)
    Random effects: random intercept for each participant ('uuid') to account for repeated measures.

    Returns the fitted model object (statsmodels result). If the mixed model fails to converge,
    falls back to an OLS with cluster-robust standard errors by participant.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    formula = (
        "LogSpeed ~ reader_view * dyslexia_bin + num_words_c + Flesch_Kincaid_c + age_c "
        "+ english_native_bin + retake_trial + C(device) + C(page_id)"
    )

    # Primary: mixed effects model with random intercepts for participants
    try:
        md = smf.mixedlm(formula, data=df, groups=df['uuid'], re_formula='1')
        mdf = md.fit(reml=False, method='lbfgs')
        return mdf

    except Exception as e:
        # Fallback: OLS with cluster-robust SEs by participant
        print('MixedLM failed (fallback to OLS with cluster-robust SEs). Error:', e)
        ols_md = smf.ols(formula, data=df).fit()
        # attach clustered covariance
        try:
            clustered = ols_md.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
            return clustered
        except Exception:
            # If clustering also fails, return the plain OLS
            return ols_md


