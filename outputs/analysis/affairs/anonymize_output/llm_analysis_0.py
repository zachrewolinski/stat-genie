from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
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
    Transform the raw Fair (Psychology Today) dataset into analysis-ready columns.

    Inputs expected: columns named feature1..feature10 as in the provided schema
    Outputs (added/renamed columns):
      - AffairCount (numeric) -- original feature2
      - AffairAny (binary) -- 1 if AffairCount > 0, else 0
      - LogAffairCount (numeric) -- log(AffairCount) for AffairCount>0, else NaN
      - HasChildren (binary) -- 1 if children present, 0 if not
      - Gender_Female (binary) -- 1 if female, 0 if male
      - Age, YearsMarried, Religiousness, Education, Occupation, MaritalHappiness (numeric)

    Rows with missing values in the variables used for the principal models are dropped.
    """

    # Rename relevant columns to meaningful names
    rename_map = {
        'feature2': 'AffairCount',
        'feature3': 'Gender',
        'feature4': 'Age',
        'feature5': 'YearsMarried',
        'feature6': 'Children',
        'feature7': 'Religiousness',
        'feature8': 'Education',
        'feature9': 'Occupation',
        'feature10': 'MaritalHappiness'
    }
    df = df.rename(columns=rename_map)

    # Define the exact required final columns (per contract)
    required_for_models = [
        'AffairCount', 'AffairAny', 'HasChildren', 'Gender_Female', 'Age', 'YearsMarried',
        'Religiousness', 'Education', 'Occupation', 'MaritalHappiness'
    ]

    # Ensure all required columns exist in the dataframe (create as NaN if missing).
    # This guarantees the transform returns a dataframe with the required column names.
    for col in required_for_models:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure numeric columns are numeric (coerce non-numeric to NaN)
    num_cols = ['AffairCount', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'MaritalHappiness']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create binary indicator for any affair
    # Preserve NaN where AffairCount is missing
    if 'AffairCount' in df.columns:
        affaircount = df['AffairCount']
        affair_any = np.where(affaircount.isna(), np.nan, (affaircount > 0).astype(int))
        df['AffairAny'] = pd.Series(affair_any, index=df.index)
    else:
        df['AffairAny'] = np.nan

    # Log of positive counts for second-stage modeling (NaN for zeros/nonpositive/missing)
    df['LogAffairCount'] = np.where(df['AffairCount'] > 0, np.log(df['AffairCount']), np.nan)

    # Standardize / binarize children indicator
    if 'Children' in df.columns:
        # map various textual representations to lower-case then to binary
        children_series = df['Children'].astype(str).str.strip().str.lower()
        mapped = children_series.map({'yes': 1, 'no': 0, 'y': 1, 'n': 0})
        df['HasChildren'] = mapped
        # if there are alternative codings (like 1/0), attempt numeric coercion for those
        mask_na = df['HasChildren'].isna()
        if mask_na.any():
            coerced = pd.to_numeric(df.loc[mask_na, 'Children'], errors='coerce')
            df.loc[mask_na, 'HasChildren'] = coerced
    else:
        # If original 'Children' column absent, HasChildren was already created above as NaN
        pass

    # Coerce HasChildren to numeric (will remain NA where unknown)
    df['HasChildren'] = pd.to_numeric(df['HasChildren'], errors='coerce')

    # Gender: create Female indicator (1 = female, 0 = male)
    if 'Gender' in df.columns:
        gender_s = df['Gender'].astype(str).str.strip().str.lower()
        mapped_g = gender_s.map({'female': 1, 'male': 0, 'f': 1, 'm': 0})
        df['Gender_Female'] = mapped_g
        # If gender encoded differently (e.g., 1/0) try coercion for the remaining NAs
        mask_g_na = df['Gender_Female'].isna()
        if mask_g_na.any():
            coerced_g = pd.to_numeric(df.loc[mask_g_na, 'Gender'], errors='coerce')
            df.loc[mask_g_na, 'Gender_Female'] = coerced_g
    else:
        # If original 'Gender' column absent, Gender_Female was already created above as NaN
        pass

    df['Gender_Female'] = pd.to_numeric(df['Gender_Female'], errors='coerce')

    # Now drop any rows that are missing any of the required modeling variables
    # (this enforces "Rows with missing values in the variables used for the principal models are dropped.")
    df = df.dropna(subset=required_for_models)

    # Ensure final columns are standard numpy-backed dtypes (not pandas nullable dtypes)
    # Cast binary indicators to float (they are guaranteed non-missing after dropna) to avoid pandas extension dtypes
    df['AffairAny'] = df['AffairAny'].astype(float)
    df['HasChildren'] = df['HasChildren'].astype(float)
    df['Gender_Female'] = df['Gender_Female'].astype(float)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Run a two-part analysis to estimate the association between having children and extramarital affairs.

    1) Logistic regression for probability of any affair (AffairAny ~ HasChildren + controls)
    2) Linear OLS on log(AffairCount) among those with AffairCount>0 (LogAffairCount ~ HasChildren + controls)

    Returns a dictionary with fitted model result objects for both parts.
    """

    # Define control variables to include
    controls = [
        'Gender_Female',
        'Age',
        'YearsMarried',
        'Religiousness',
        'Education',
        'Occupation',
        'MaritalHappiness'
    ]

    # Ensure required columns are present
    required_cols = ['AffairAny', 'HasChildren'] + controls
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Prepare design matrix for logistic regression (any affair)
    X_logit = df[['HasChildren'] + controls].copy()
    # Ensure numeric dtype for design matrix
    X_logit = X_logit.apply(pd.to_numeric, errors='coerce').astype(float)
    X_logit = sm.add_constant(X_logit, has_constant='add')
    y_logit = pd.to_numeric(df['AffairAny'], errors='coerce').astype(float)

    # Align and drop any rows with missing data for the logistic model
    data_logit = pd.concat([X_logit, y_logit.rename('AffairAny')], axis=1)
    data_logit = data_logit.dropna()
    if data_logit.shape[0] == 0:
        # No complete cases: return None results rather than raising an error
        return {
            'logit_model': None,
            'ols_positive_model': None
        }
    y_logit_clean = data_logit['AffairAny']
    X_logit_clean = data_logit.drop(columns=['AffairAny'])

    # Fit logistic regression (probability of any affair)
    logit_model = None
    try:
        logit_model = sm.Logit(y_logit_clean, X_logit_clean).fit(disp=False)
    except Exception:
        # If Logit fails to converge or raises, fall back to GLM with binomial family
        try:
            logit_model = sm.GLM(y_logit_clean, X_logit_clean, family=sm.families.Binomial()).fit()
        except Exception:
            # If fallback also fails, leave as None
            logit_model = None

    # Second-stage model: among respondents with AffairCount > 0
    # Ensure LogAffairCount exists in dataframe (transform guarantees it)
    df_pos = df[df['AffairCount'] > 0].copy()
    results_pos = None
    if len(df_pos) >= (len(controls) + 2):
        X_pos = df_pos[['HasChildren'] + controls].copy()
        X_pos = X_pos.apply(pd.to_numeric, errors='coerce').astype(float)
        X_pos = sm.add_constant(X_pos, has_constant='add')
        y_pos = pd.to_numeric(df_pos['LogAffairCount'], errors='coerce').astype(float)
        # Align and drop missing for OLS
        data_pos = pd.concat([X_pos, y_pos.rename('LogAffairCount')], axis=1)
        data_pos = data_pos.dropna()
        if data_pos.shape[0] >= (len(controls) + 2):
            y_pos_clean = data_pos['LogAffairCount']
            X_pos_clean = data_pos.drop(columns=['LogAffairCount'])
            # Fit OLS on the log-transformed positive counts
            try:
                ols_model = sm.OLS(y_pos_clean, X_pos_clean).fit()
                results_pos = ols_model
            except Exception:
                results_pos = None
        else:
            results_pos = None
    else:
        # Not enough positive observations to estimate second-stage model
        results_pos = None

    # Return both fitted results
    return {
        'logit_model': logit_model,
        'ols_positive_model': results_pos
    }