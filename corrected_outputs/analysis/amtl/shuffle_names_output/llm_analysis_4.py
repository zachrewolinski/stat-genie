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
    Transform the raw dataset into a dataframe suitable for binomial modeling of AMTL.

    Assumptions (based on the provided but inconsistent schema):
    - 'stdev_age' contains the count of missing teeth for the given tooth class (per-row successes).
    - 'sockets' contains the number of observable tooth sockets (trials) for that row.
    - 'num_amtl' contains estimated age-at-death (continuous) for the specimen.
    - 'pop' contains an estimate of sex as probability male (0-1). If 'pop' is missing but another sex column exists it will be attempted.
    - The column that actually contains the taxonomic genus (Homo / Pan / Pongo / Papio) appears in the provided schema as 'age'. If 'age' does not contain genus labels, the code will try 'genus'.
    - Tooth class (Anterior/Posterior/Premolar) may be in the 'genus' column per schema inconsistencies; the code checks for common tooth-class labels and uses whichever column contains them.

    The function returns the dataframe with the following NEW/clean columns used in modeling:
      - AMTL_count (int): number of missing teeth (successes)
      - Sockets (float/int): number of observable sockets (trials)
      - AMTL_prop (float): AMTL_count / Sockets
      - AgeAtDeath (float): estimated age at death
      - MaleProb (float): probability specimen is male (0-1)
      - GenusClean (str): cleaned genus label (Homo, Pan, Pongo, Papio, or other)
      - IsHuman (int): 1 if GenusClean contains 'Homo' (case-insensitive), else 0
      - ToothClass (category): tooth class standardized to anterior/premolar/posterior/other
    """

    df = df.copy()

    # Standardize column names existence
    # Specimen ID
    if 'specimen' in df.columns:
        df['Specimen'] = df['specimen']
    
    # Determine where genus (taxon) information is stored. Prefer 'age' if it contains genus names,
    # otherwise fall back to the 'genus' column.
    genus_col = None
    if 'age' in df.columns:
        try:
            if df['age'].dropna().astype(str).str.contains(r'\b(Homo|Pan|Pongo|Papio)\b', case=False, regex=True).any():
                genus_col = 'age'
        except Exception:
            genus_col = None
    if genus_col is None and 'genus' in df.columns:
        try:
            if df['genus'].dropna().astype(str).str.contains(r'\b(Homo|Pan|Pongo|Papio)\b', case=False, regex=True).any():
                genus_col = 'genus'
        except Exception:
            genus_col = None
    # If still unknown, default to 'age' if exists, else 'genus' if exists
    if genus_col is None:
        if 'age' in df.columns:
            genus_col = 'age'
        elif 'genus' in df.columns:
            genus_col = 'genus'

    # Create cleaned genus column
    if genus_col is not None:
        df['GenusClean'] = df[genus_col].astype(str).str.strip().replace({'nan': None})
    else:
        df['GenusClean'] = None

    # Standardize tooth-class. Many schema labels are inconsistent; detect common tooth-class labels in 'genus' or 'tooth_class'.
    tooth_src = None
    if 'genus' in df.columns:
        # If genus column includes anterior/posterior/premolar, use it as tooth class
        try:
            if df['genus'].dropna().astype(str).str.contains(r'(?i)anterior|posterior|premolar').any():
                tooth_src = 'genus'
        except Exception:
            tooth_src = None
    if tooth_src is None and 'tooth_class' in df.columns:
        tooth_src = 'tooth_class'
    if tooth_src is None:
        # fallback: create a generic 'other'
        df['ToothClass'] = 'other'
    else:
        df['ToothClass'] = df[tooth_src].astype(str).str.strip()

    # Normalize ToothClass into a small set
    def _standardize_tooth_class(x):
        if pd.isna(x):
            return 'other'
        xlow = str(x).lower()
        if 'anterior' in xlow:
            return 'anterior'
        if 'premolar' in xlow:
            return 'premolar'
        if 'posterior' in xlow:
            return 'posterior'
        # sometimes 'molar' terms or others; keep as 'other'
        return 'other'

    df['ToothClass'] = df['ToothClass'].apply(_standardize_tooth_class).astype('category')

    # Create AMTL_count (successes) and Sockets (trials).
    # Based on schema inconsistencies, 'stdev_age' is interpreted here as the count of missing teeth for the row.
    if 'stdev_age' in df.columns:
        df['AMTL_count'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    elif 'num_amtl' in df.columns:
        # fallback if the other mapping is different; attempt to use it
        df['AMTL_count'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    else:
        df['AMTL_count'] = np.nan

    # Sockets: number of observable sockets
    if 'sockets' in df.columns:
        df['Sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    elif 'prob_male' in df.columns:
        # highly unlikely fallback; but keep code robust
        df['Sockets'] = pd.to_numeric(df['prob_male'], errors='coerce')
    else:
        df['Sockets'] = np.nan

    # Age at death (continuous)
    if 'num_amtl' in df.columns:
        df['AgeAtDeath'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    elif 'stdev_age' in df.columns:
        # if labels swapped, try the other
        df['AgeAtDeath'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    else:
        df['AgeAtDeath'] = np.nan

    # Male probability (0-1). Prefer 'pop' (schema suggests it is 0-1). If missing, try other columns heuristically.
    if 'pop' in df.columns:
        df['MaleProb'] = pd.to_numeric(df['pop'], errors='coerce')
    elif 'prob_male' in df.columns:
        # If 'prob_male' appears to be counts (range >1), try to scale to 0-1 if maximum >1.
        tmp = pd.to_numeric(df['prob_male'], errors='coerce')
        if tmp.dropna().max() > 1:
            # scale to [0,1] by dividing by max observed (best-effort)
            df['MaleProb'] = tmp / tmp.max()
        else:
            df['MaleProb'] = tmp
    else:
        df['MaleProb'] = np.nan

    # Clean numeric AMTL_count and Sockets: round AMTL_count to nearest integer, ensure non-negative
    df['AMTL_count'] = pd.to_numeric(df['AMTL_count'], errors='coerce')
    df['AMTL_count'] = df['AMTL_count'].round().clip(lower=0)
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce')

    # Remove rows with non-positive sockets or missing essential fields
    df = df.dropna(subset=['AMTL_count', 'Sockets', 'GenusClean', 'AgeAtDeath'])
    df = df[df['Sockets'] > 0]

    # Ensure AMTL_count does not exceed Sockets
    # If it does (possible because of schema confusion), cap AMTL_count at Sockets
    df['AMTL_count'] = np.minimum(df['AMTL_count'], df['Sockets'])

    # Proportion for modeling
    df['AMTL_prop'] = df['AMTL_count'] / df['Sockets']

    # Create binary human indicator
    df['GenusClean'] = df['GenusClean'].astype(str).str.replace('_', ' ').str.strip()
    df['IsHuman'] = df['GenusClean'].str.contains(r'(?i)homo').fillna(False).astype(int)

    # Final required columns: ensure types
    df['AgeAtDeath'] = pd.to_numeric(df['AgeAtDeath'], errors='coerce')
    df['MaleProb'] = pd.to_numeric(df['MaleProb'], errors='coerce')

    # Keep only rows with non-missing modeling variables
    df = df.dropna(subset=['AMTL_count', 'Sockets', 'AMTL_prop', 'IsHuman', 'AgeAtDeath'])

    # Make ToothClass categorical with a stable ordering
    df['ToothClass'] = pd.Categorical(df['ToothClass'], categories=['anterior', 'premolar', 'posterior', 'other'])

    # Return transformed df including original specimen id if present
    keep_cols = ['Specimen', 'GenusClean', 'IsHuman', 'AMTL_count', 'Sockets', 'AMTL_prop', 'AgeAtDeath', 'MaleProb', 'ToothClass']
    # Add columns that might not exist but are useful
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a binomial (logistic) GLM to test whether modern humans have higher AMTL than non-human primates,
    controlling for age, sex, and tooth class.

    Model specification (generalized linear model, binomial family):
      AMTL_prop ~ IsHuman + AgeAtDeath + MaleProb + C(ToothClass)
    with frequency weights equal to the number of sockets (so the model is effectively on counts: AMTL_count out of Sockets).

    Returns the fitted GLM results object (statsmodels).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['AMTL_prop', 'Sockets', 'IsHuman', 'AgeAtDeath', 'MaleProb', 'ToothClass']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop rows with any remaining missing predictor values (GLM can't handle NaNs here)
    model_df = df.dropna(subset=['AMTL_prop', 'Sockets', 'IsHuman', 'AgeAtDeath', 'MaleProb', 'ToothClass']).copy()

    # Formula: treat ToothClass as categorical. Use IsHuman as the primary predictor (binary).
    formula = 'AMTL_prop ~ IsHuman + AgeAtDeath + MaleProb + C(ToothClass)'

    # Fit GLM binomial with weights equal to number of sockets (so the response is successes/trials)
    # Note: Using freq_weights is a convenient way to pass the number of trials when the response is a proportion.
    glm_binom = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial(), freq_weights=model_df['Sockets'])
    results = glm_binom.fit()

    # Print a brief summary to console for user inspection (caller can inspect returned results object for details)
    print(results.summary())

    return results


