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
    Transform the raw Fair (Psychology Today) dataset into a dataframe with the columns used by the model.

    Required input columns (from schema):
      - feature2: numeric frequency of extramarital intercourse in past year (0 = none, other codes for frequencies)
      - feature3: gender ('male'/'female')
      - feature4: age coding (numeric)
      - feature5: years married (numeric)
      - feature6: indicator for children in marriage ('yes'/'no')
      - feature7: religiosity (1-5)
      - feature8: education coding (numeric)
      - feature9: occupation score (numeric)
      - feature10: self rating of marriage (1-5)

    Produces columns:
      - AffairFreq: numeric frequency (same as feature2)
      - AnyAffair: binary indicator (1 if AffairFreq > 0, else 0)
      - LogAffairFreqPos: log(AffairFreq) for rows with AffairFreq > 0, NaN otherwise
      - AffairFreqPos: numeric frequency for rows with AffairFreq > 0, NaN otherwise
      - HasChildren: 1 if feature6 == 'yes', 0 if 'no'
      - Gender_Male, Age, YearsMarried, Religiosity, Education, Occupation, MaritalHappiness

    Notes:
      - Rows with missing values in the main variables are dropped.
      - We do not recode the special codes for feature2 beyond using the numeric values provided (e.g., 7, 12);
        these values represent ordinal/frequency codes in the original questionnaire and are treated as numeric frequency indicators here.
    """
    # copy to avoid modifying in place
    df = df.copy()

    # Keep only the columns we need (if present). This will raise a KeyError if required columns are missing.
    required = ['feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    # Rename / derive clean variable names used in modeling
    # Affair frequency (dependent variable). Keep as numeric as provided.
    df['AffairFreq'] = pd.to_numeric(df['feature2'], errors='coerce')

    # Binary indicator: any affair (1 if frequency > 0)
    df['AnyAffair'] = (df['AffairFreq'] > 0).astype(int)

    # Positive-frequency subset variable and log transform for intensity model
    df['AffairFreqPos'] = df['AffairFreq'].where(df['AffairFreq'] > 0, np.nan)
    # Use natural log of the reported frequency for positive cases (log scale helps skew)
    df['LogAffairFreqPos'] = np.log(df['AffairFreqPos'])

    # Independent variable: children in marriage
    # Accept common encodings: 'yes'/'no' strings (case-insensitive), 1/0, or boolean
    def map_children(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x_low = x.strip().lower()
            if x_low in ['yes','y','true','t','1']:
                return 1
            if x_low in ['no','n','false','f','0']:
                return 0
        try:
            # numeric
            xv = float(x)
            if xv == 1:
                return 1
            if xv == 0:
                return 0
        except Exception:
            pass
        return np.nan

    df['HasChildren'] = df['feature6'].apply(map_children).astype('float')

    # Controls
    # Gender -> male indicator
    def map_male(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in ['male','m']:
                return 1
            if xl in ['female','f']:
                return 0
        try:
            xv = float(x)
            # If coded numerically (e.g., 0/1) we assume 1 means male if sample suggests that.
            return 1 if xv == 1 else 0
        except Exception:
            return np.nan

    df['Gender_Male'] = df['feature3'].apply(map_male).astype('float')

    # Direct mappings for numeric controls (coerce to numeric and keep NaN where invalid)
    df['Age'] = pd.to_numeric(df['feature4'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['Religiosity'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Education'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['MaritalHappiness'] = pd.to_numeric(df['feature10'], errors='coerce')

    # Drop rows missing the primary variables needed for modeling: AffairFreq and HasChildren and core controls
    # We will keep rows that have AffairFreq and HasChildren; models will drop additional NAs as needed.
    df = df.dropna(subset=['AffairFreq','HasChildren'])

    # For convenience, cast HasChildren and AnyAffair and Gender_Male to integer types where not null
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['AnyAffair'] = df['AnyAffair'].astype(int)
    # Gender may have missing values; leave as float with NaN if missing

    # Return transformed df with the new variables appended (original columns retained)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run a two-part analysis to answer whether having children decreases engagement in extramarital affairs.

    1) Binary (extensive margin): Logistic regression predicting AnyAffair (0/1) from HasChildren and controls.
    2) Intensive margin (conditional on having any affair): OLS predicting log-frequency (LogAffairFreqPos) among respondents with AffairFreq > 0.

    Returns a dict with the fitted model results objects.
    """
    results = {}

    # Ensure required columns exist
    required_cols = ['AnyAffair','HasChildren','Gender_Male','Age','YearsMarried','Religiosity','Education','Occupation','MaritalHappiness','LogAffairFreqPos']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Prepare design matrix
    controls = ['Gender_Male','Age','YearsMarried','Religiosity','Education','Occupation','MaritalHappiness']
    # Common formula string (we will build matrices manually for statsmodels)
    X = df[['HasChildren'] + controls].copy()
    X = sm.add_constant(X, has_constant='add')

    # 1) Logistic regression for AnyAffair
    y_bin = df['AnyAffair']
    # Drop rows with NA in X or y
    bin_mask = X.notnull().all(axis=1) & y_bin.notnull()
    X_bin = X.loc[bin_mask]
    y_bin = y_bin.loc[bin_mask]

    if len(y_bin.unique()) == 1:
        # Edge case: no variation in dependent variable
        results['logit'] = None
        results['logit_message'] = 'No variation in AnyAffair; logistic regression not estimated.'
    else:
        logit_model = sm.Logit(y_bin, X_bin)
        try:
            logit_res = logit_model.fit(disp=False)
        except Exception:
            # fallback to GLM Binomial (more stable in some cases)
            logit_model = sm.GLM(y_bin, X_bin, family=sm.families.Binomial())
            logit_res = logit_model.fit()
        results['logit'] = logit_res

    # 2) OLS on log-frequency among those with positive frequency
    pos_mask = df['AffairFreq'] > 0
    df_pos = df.loc[pos_mask].copy()
    # Drop rows with missing LogAffairFreqPos or controls
    df_pos = df_pos.dropna(subset=['LogAffairFreqPos'] + controls + ['HasChildren'])

    if df_pos.shape[0] < 10:
        results['ols'] = None
        results['ols_message'] = 'Too few positive-affair observations to reliably estimate OLS.'
    else:
        X_pos = df_pos[['HasChildren'] + controls]
        X_pos = sm.add_constant(X_pos, has_constant='add')
        y_pos = df_pos['LogAffairFreqPos']
        ols_model = sm.OLS(y_pos, X_pos)
        ols_res = ols_model.fit()
        results['ols'] = ols_res

    # Provide a simple aggregated summary: mean affair rates by HasChildren
    summary_table = df.groupby('HasChildren').agg(
        N=('AffairFreq','size'),
        AnyAffairRate=('AnyAffair','mean'),
        MeanAffairFreq=('AffairFreq','mean')
    ).reset_index()
    results['summary_table'] = summary_table

    return results


