from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/shuffle_names_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the raw dataset into the final analysis dataframe.

    Input columns expected (based on provided schema):
      - 'num_amtl' : number of missing teeth (AMTL) for the record (successes)
      - 'sockets'  : number of observable tooth sockets that could be scored (trials)
      - 'genus'    : specimen genus (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio')
      - 'age'      : estimated age at death (numeric if present)
      - 'pop' or 'prob_male' : numeric estimate of sex (probability male) or binary sex indicator
      - 'tooth_class' : tooth class (e.g., 'Anterior', 'Posterior', 'Premolar')

    Produces these columns used by the model:
      - 'AMTL_successes' (numeric)
      - 'AMTL_trials' (numeric)
      - 'AMTL_rate' (proportion successes/trials)
      - 'IsHuman' (0/1)
      - 'Age' (numeric)
      - 'IsMale' (0/1)
      - 'tooth_class' (categorical)
    """
    # Work on a copy
    df = df.copy()

    # 1) Standardize successes and trials
    df['AMTL_successes'] = pd.to_numeric(df.get('num_amtl', df.get('num_amtl', None)), errors='coerce')
    df['AMTL_trials'] = pd.to_numeric(df.get('sockets', df.get('sockets', None)), errors='coerce')

    # Drop rows where we can't compute a binomial outcome
    df = df.dropna(subset=['AMTL_successes', 'AMTL_trials'])

    # Remove nonsensical trials (<=0)
    df = df[df['AMTL_trials'] > 0]

    # If successes are not integers or out of bounds, coerce sensibly: round to nearest and clamp
    df['AMTL_successes'] = pd.to_numeric(df['AMTL_successes'], errors='coerce')
    # Round to nearest integer
    df['AMTL_successes'] = df['AMTL_successes'].round().astype(float)
    # Clamp successes between 0 and trials
    df.loc[df['AMTL_successes'] < 0, 'AMTL_successes'] = 0.0
    df.loc[df['AMTL_successes'] > df['AMTL_trials'], 'AMTL_successes'] = df.loc[df['AMTL_successes'] > df['AMTL_trials'], 'AMTL_trials']

    # Proportion
    df['AMTL_rate'] = df['AMTL_successes'] / df['AMTL_trials']

    # 2) Create IsHuman indicator from 'genus'
    # We treat any genus value containing 'Homo' (case-insensitive) as human
    df['IsHuman'] = df['genus'].astype(str).str.contains('Homo', case=False, na=False).astype(int)

    # 3) Age: try to coerce provided 'age' column to numeric
    df['Age'] = pd.to_numeric(df.get('age'), errors='coerce')
    # If Age is entirely missing, attempt plausible fallbacks (use stdev_age only as a last resort)
    if df['Age'].isna().all():
        # try 'num_amtl' is unlikely to be age, but try 'stdev_age' (schema mismatch possible)
        df['Age'] = pd.to_numeric(df.get('stdev_age'), errors='coerce')
    # If still missing values, impute median age (simple, transparent approach)
    if df['Age'].isna().any():
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)

    # 4) Sex: derive IsMale from 'pop' (0-1 probability) or 'prob_male' if present
    # Prefer 'pop' if it looks like a probability in [0,1]
    male_col = None
    if 'pop' in df.columns:
        male_col = 'pop'
    elif 'prob_male' in df.columns:
        male_col = 'prob_male'

    df['MaleProb_raw'] = np.nan
    if male_col is not None:
        # coerce to numeric
        df['MaleProb_raw'] = pd.to_numeric(df[male_col], errors='coerce')

    # If MaleProb_raw looks like 0/1 or probabilities, create IsMale; otherwise try to parse 'M'/'F'
    # If still missing, impute 0.5 (unknown)
    # Case: if values >1 but only two unique values {1,2} maybe coded; map 1->male, 2->female? Hard to know.
    # Strategy: if values are in [0,1] use threshold 0.5; if values are integer and only {0,1} use as-is; if values are integer and {1,2} assume 1=male, 2=female.
    mp = df['MaleProb_raw']
    is_finite = mp.notna()
    if is_finite.any():
        unique_vals = pd.Series(mp[is_finite].unique())
        # normalize
        if ((mp >= 0) & (mp <= 1)).all():
            df['IsMale'] = (mp >= 0.5).astype(int)
        else:
            # integer-coded possibilities
            unique_ints = unique_vals.dropna().astype(float)
            if set(unique_ints.dropna().astype(int).unique()).issubset({0,1}):
                df['IsMale'] = mp.astype(int)
            elif set(unique_ints.dropna().astype(int).unique()).issubset({1,2}):
                # assume 1=male,2=female
                df['IsMale'] = (mp.astype(int) == 1).astype(int)
            else:
                # fallback: treat values > 0.5 as male
                df['IsMale'] = (mp >= 0.5).astype(int)
    else:
        # Try textual sex columns (not in schema but safe): 'sex'
        if 'sex' in df.columns:
            s = df['sex'].astype(str).str.strip().str.lower()
            df['IsMale'] = s.isin(['m', 'male', '1']).astype(int)
        else:
            # no sex info: set to 0.5 (unknown) and then map to 0/1 by threshold -> keeps variance small
            df['IsMale'] = 0.5

    # If IsMale has non-integer (0.5) entries, convert to numeric and fill to nearest integer by threshold
    if df['IsMale'].dtype != int and df['IsMale'].dtype != float:
        df['IsMale'] = pd.to_numeric(df['IsMale'], errors='coerce').fillna(0)
    # If still contains 0.5 (imputed unknown), set to 0 (conservative) or better: keep as 0.5 but GLM expects numeric covariate - allow 0.5
    # We'll keep it as numeric (0,1 or 0.5) because it represents uncertainty.

    # 5) Tooth class: ensure categorical with expected levels
    df['tooth_class'] = df.get('tooth_class', df.get('genus', pd.Series(['Unknown'] * len(df)))).astype(str)
    # Common canonicalization
    df['tooth_class'] = df['tooth_class'].str.strip().str.capitalize()

    # 6) Final drop of rows with any required missing values
    required = ['AMTL_successes', 'AMTL_trials', 'AMTL_rate', 'IsHuman', 'Age', 'IsMale', 'tooth_class']
    df = df.dropna(subset=required)

    # Keep only columns needed for modeling plus original identifiers
    keep_cols = ['specimen', 'genus', 'AMTL_successes', 'AMTL_trials', 'AMTL_rate', 'IsHuman', 'Age', 'IsMale', 'tooth_class']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (GLM) model for AMTL frequency comparing humans vs non-human primates,
    controlling for age, sex, and tooth class.

    The model treats AMTL as a binomial outcome (proportion with weights equal to number of sockets):
      AMTL_rate ~ IsHuman + Age + IsMale + C(tooth_class)
    with freq_weights = AMTL_trials.

    Returns the fitted GLM results object (statsmodels.genmod.generalized_linear_model.GLMResults).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure columns exist
    assert 'AMTL_rate' in df.columns and 'AMTL_trials' in df.columns, "Transformed dataframe must contain AMTL_rate and AMTL_trials"

    # Formula: proportion outcome modeled with binomial family and weights equal to trials
    formula = 'AMTL_rate ~ IsHuman + Age + IsMale + C(tooth_class)'

    # Fit GLM with binomial family using freq_weights to pass the number of trials
    # (modeling proportions with binomial denominator)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['AMTL_trials'])
    results = model.fit()

    # Recommended: check for overdispersion after fitting; user can examine results.summary() and residuals
    # Return the full results object so downstream code can inspect coefficients, CIs, etc.
    return results


