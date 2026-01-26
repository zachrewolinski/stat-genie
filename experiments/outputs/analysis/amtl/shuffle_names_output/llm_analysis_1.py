from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/shuffle_names_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a cleaned dataframe ready for binomial modeling of AMTL.

    Expected behavior / mapping (robust to slight schema mismatches):
    - AMTL_count: number of missing teeth for the given specimen/tooth-class record. Mapped from 'stdev_age' when present.
    - Sockets: number of observable sockets (trials). Prefer 'prob_male' if it contains integer socket counts; fall back to 'sockets' if needed.
    - Age_at_death: continuous age estimate mapped from 'num_amtl' (per dataset description alignment).
    - ProbMale: continuous 0-1 sex estimate mapped from 'pop'.
    - Genus: taxonomic genus (Homo, Pan, Pongo, Papio) taken from column 'age' (per dataset description mapping).
    - ToothClass: tooth class (Anterior, Posterior, Premolar) taken from column 'genus' (per dataset description mapping).

    The function will:
    - parse numeric fields robustly
    - round/cast counts to integers
    - drop rows with missing critical fields
    - create AMTL_prop, IsHomo, SexMale, and standardized age
    - normalize ToothClass categories and mark unknown classes as 'Other'
    """
    df = df.copy()

    # --- Robust column mapping based on field names and provided descriptions ---
    # Map candidate columns to the conceptual variables. The dataset schema appears to have
    # mismatched descriptions; use column names but be defensive.
    # AMTL count (number of missing teeth) -> prefer 'stdev_age'
    if 'stdev_age' in df.columns:
        df['AMTL_count'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    else:
        # fallback: try 'num_amtl' (if dataset uses that name for counts)
        df['AMTL_count'] = pd.to_numeric(df.get('num_amtl', np.nan), errors='coerce')

    # Sockets (number of observable sockets) -> prefer 'prob_male' (schema suggests this holds socket counts)
    if 'prob_male' in df.columns:
        df['Sockets'] = pd.to_numeric(df['prob_male'], errors='coerce')
    elif 'sockets' in df.columns:
        df['Sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    else:
        df['Sockets'] = np.nan

    # Age at death (continuous) -> prefer 'num_amtl' per schema mapping
    if 'num_amtl' in df.columns:
        df['Age_at_death'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    elif 'stdev_age' in df.columns:
        # if no explicit age column, keep missing
        df['Age_at_death'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    else:
        df['Age_at_death'] = np.nan

    # Sex probability (0-1) -> 'pop'
    df['ProbMale'] = pd.to_numeric(df.get('pop', np.nan), errors='coerce')

    # Genus (taxon) -> many datasets store genus under 'age' according to schema
    if 'age' in df.columns:
        df['Genus'] = df['age'].astype(str).str.strip()
    elif 'genus' in df.columns:
        df['Genus'] = df['genus'].astype(str).str.strip()
    else:
        df['Genus'] = df.get('genus', '').astype(str).str.strip()

    # Tooth class -> schema indicates 'genus' may actually contain tooth class labels
    if 'genus' in df.columns:
        df['ToothClass'] = df['genus'].astype(str).str.strip()
    elif 'tooth_class' in df.columns:
        df['ToothClass'] = df['tooth_class'].astype(str).str.strip()
    else:
        df['ToothClass'] = df.get('tooth_class', '').astype(str).str.strip()

    # Specimen identifier
    df['SpecimenID'] = df.get('specimen', df.index).astype(str)

    # --- Clean and validate numeric counts ---
    # Drop rows missing critical fields
    df = df.dropna(subset=['AMTL_count', 'Sockets', 'Genus', 'ToothClass'])

    # Round counts to integers and ensure Sockets > 0
    df['AMTL_count'] = pd.to_numeric(df['AMTL_count'], errors='coerce').round().astype('Int64')
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce').round().astype('Int64')
    df = df[df['Sockets'].notna()]
    df = df[df['AMTL_count'].notna()]
    df = df[df['Sockets'] > 0]

    # Clip AMTL_count to [0, Sockets]
    df['AMTL_count'] = df.apply(lambda r: int(max(0, min(r['AMTL_count'], r['Sockets']))), axis=1)

    # Compute proportion for inspection
    df['AMTL_prop'] = df['AMTL_count'] / df['Sockets']

    # --- Genus indicator for Homo sapiens ---
    df['IsHomo'] = df['Genus'].str.contains('homo', case=False, na=False).astype(int)

    # --- Age standardization for model stability ---
    df['Age_at_death'] = pd.to_numeric(df['Age_at_death'], errors='coerce')
    # If Age_at_death is completely missing, Age_std will be NaN (model will handle or we can drop)
    if df['Age_at_death'].notna().sum() > 1:
        df['Age_std'] = (df['Age_at_death'] - df['Age_at_death'].mean()) / (df['Age_at_death'].std(ddof=0) if df['Age_at_death'].std(ddof=0) != 0 else 1.0)
    else:
        df['Age_std'] = df['Age_at_death']

    # --- Sex variables ---
    df['ProbMale'] = pd.to_numeric(df['ProbMale'], errors='coerce')
    # Clip to [0,1] where applicable
    df.loc[df['ProbMale'].notna(), 'ProbMale'] = df.loc[df['ProbMale'].notna(), 'ProbMale'].clip(0, 1)
    df['SexMale'] = (df['ProbMale'] >= 0.5).astype(int)

    # --- Normalize ToothClass categories ---
    # Standardize string to lower-case and detect keywords
    def normalize_tooth(tc):
        if pd.isna(tc) or tc == '':
            return 'Other'
        s = str(tc).lower()
        if 'ante' in s or 'incis' in s or 'canin' in s:
            return 'Anterior'
        if 'post' in s or 'molar' in s:
            return 'Posterior'
        if 'prem' in s:
            return 'Premolar'
        # if exact matches
        if s in ['anterior', 'posterior', 'premolar']:
            return s.capitalize()
        return 'Other'

    df['ToothClass'] = df['ToothClass'].apply(normalize_tooth)

    # Keep only records for which ToothClass is one of the known classes or 'Other' (we'll include dummies for known classes)
    df['ToothClass'] = df['ToothClass'].astype(str)

    # Final: drop rows with any remaining NA in key model inputs
    key_cols = ['AMTL_count', 'Sockets', 'IsHomo', 'Age_std', 'ProbMale', 'ToothClass', 'SpecimenID']
    # Age_std or ProbMale may be partially missing; we will allow rows with missing Age_std or ProbMale but recommend dropping them for the main model.
    # For safety, only drop rows with AMTL_count or Sockets missing (we already did), keep others and let model drop if necessary.

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial regression to test whether modern humans (IsHomo) have higher AMTL frequency
    after controlling for age, sex, and tooth class.

    Modeling approach:
    - Use a Binomial GLM on counts: endog is a 2-column array [AMTL_count, Sockets - AMTL_count]
      which is supported by statsmodels' GLM for Binomial.
    - Predictors: IsHomo (primary IV), Age_std (continuous), ProbMale (continuous), SexMale (binary),
      and dummy variables for ToothClass (Posterior and Premolar; Anterior as reference).
    - Use cluster-robust standard errors clustered on SpecimenID to account for within-specimen correlation
      when multiple tooth-class observations come from the same specimen.

    Returns a dictionary with the raw GLM result and clustered-robust result (if clustering succeeds).
    """
    import statsmodels.api as sm
    import numpy as np

    df = df.copy()

    # Ensure necessary columns exist
    required = ['AMTL_count', 'Sockets', 'IsHomo', 'Age_std', 'ProbMale', 'ToothClass', 'SpecimenID']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop rows with missing counts (should be none after transform) and ensure integers
    df = df[df['Sockets'].notna()]
    df = df[df['AMTL_count'].notna()]

    # Construct outcome matrix for binomial: successes and failures
    successes = np.asarray(df['AMTL_count'], dtype=int)
    trials = np.asarray(df['Sockets'], dtype=int)
    failures = trials - successes
    # Endog for Binomial GLM can be 2-column array [successes, failures]
    y = np.column_stack((successes, failures))

    # Build design matrix
    X_base = pd.DataFrame({
        'IsHomo': df['IsHomo'].astype(int),
        'Age_std': df['Age_std'].astype(float),
        'ProbMale': df['ProbMale'].astype(float),
        'SexMale': df['SexMale'].astype(int),
    }, index=df.index)

    # ToothClass dummies: drop first (reference = Anterior). Keep 'Other' as additional category if present.
    tooth_dummies = pd.get_dummies(df['ToothClass'], prefix='Tooth', drop_first=True)
    X = pd.concat([X_base, tooth_dummies], axis=1)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit Binomial GLM
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Try to compute cluster-robust SEs by SpecimenID
    clustered_res = None
    try:
        # This returns a results instance with robust covariance
        clustered_res = res.get_robustcov_results(cov_type='cluster', groups=df['SpecimenID'])
    except Exception as e:
        # If clustering fails, keep clustered_res as None and return raw result
        clustered_res = None

    # Prepare a concise output object
    out = {
        'glm_result': res,
        'glm_clustered_result': clustered_res,
        'design_matrix_columns': X.columns.tolist()
    }

    # Print short model summaries for user inspection
    try:
        print('--- GLM (Binomial) summary (naive SE) ---')
        print(res.summary())
        if clustered_res is not None:
            print('\n--- GLM (Binomial) summary (clustered SE by SpecimenID) ---')
            print(clustered_res.summary())
        else:
            print('\nClustered SE by SpecimenID not available; see glm_result for naive SEs.')
    except Exception:
        # In non-interactive settings, summary printing may fail; ignore
        pass

    return out


