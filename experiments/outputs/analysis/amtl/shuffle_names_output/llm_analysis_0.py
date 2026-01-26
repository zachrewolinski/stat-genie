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
    Clean and harmonize the input dataset to produce the final dataframe used for modeling.

    Expected original columns (per provided schema) and how they are mapped into the final dataframe:
      - original 'genus' column actually encodes tooth class labels (e.g., 'Anterior','Posterior','Premolar')
      - original 'specimen' kept as-is
      - original 'stdev_age' contains the count of teeth missing of the given class -> mapped to num_missing
      - original 'prob_male' contains the number of observable sockets that could be scored -> mapped to n_sockets
      - original 'num_amtl' contains estimated age at death -> mapped to age_at_death
      - original 'sockets' contains assigned uncertainty of age at death -> mapped to age_sd
      - original 'pop' contains estimate/probability of being male (0-1) -> mapped to sex_male_prob
      - original 'age' contains specimen genus (e.g., 'Homo sapiens','Pan','Pongo','Papio') -> mapped to genus
      - original 'tooth_class' sometimes contains region/population info; we keep it as 'origin' if present but it is not required for the primary model.

    The function will:
      - create standardized columns: ['specimen','genus','tooth_class','num_missing','n_sockets','prop_missing','age_at_death','age_sd','sex_male_prob']
      - drop rows with invalid or missing denominator (n_sockets <= 0) or missing key fields
      - coerce types appropriately and standardize genus labels
    """
    import numpy as np
    import pandas as pd

    # Work on a copy
    df = df.copy()

    # Map ambiguous/original columns to clear names used in modeling
    # The provided schema contains mismatches between column names and descriptions. We map according to the documented descriptions.
    # original column 'genus' in schema appears to be tooth class labels
    if 'genus' in df.columns:
        df['tooth_class'] = df['genus']

    # specimen identifier
    if 'specimen' in df.columns:
        df['specimen'] = df['specimen']

    # num_missing of given tooth class: described under 'stdev_age'
    if 'stdev_age' in df.columns:
        df['num_missing'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    else:
        # fallback: if a proper 'num_missing' column already exists
        df['num_missing'] = pd.to_numeric(df.get('num_missing', np.nan), errors='coerce')

    # number of observable sockets that could be scored for missing teeth: described under 'prob_male'
    if 'prob_male' in df.columns:
        df['n_sockets'] = pd.to_numeric(df['prob_male'], errors='coerce')
    else:
        df['n_sockets'] = pd.to_numeric(df.get('n_sockets', np.nan), errors='coerce')

    # age_at_death (numeric) described under 'num_amtl' in schema
    if 'num_amtl' in df.columns:
        df['age_at_death'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    else:
        df['age_at_death'] = pd.to_numeric(df.get('age_at_death', np.nan), errors='coerce')

    # age uncertainty described under 'sockets' column in schema
    if 'sockets' in df.columns:
        df['age_sd'] = pd.to_numeric(df['sockets'], errors='coerce')
    else:
        df['age_sd'] = pd.to_numeric(df.get('age_sd', np.nan), errors='coerce')

    # sex estimate: 'pop' in schema (0-1)
    if 'pop' in df.columns:
        df['sex_male_prob'] = pd.to_numeric(df['pop'], errors='coerce')
    else:
        # if there's a column named 'sex' or 'prob_male' used differently, prefer 'sex_male_prob' if present
        df['sex_male_prob'] = pd.to_numeric(df.get('sex_male_prob', np.nan), errors='coerce')

    # genus (taxon) is stored in column named 'age' in the provided schema
    if 'age' in df.columns:
        df['genus'] = df['age'].astype(str)
    else:
        df['genus'] = df.get('genus_final', np.nan)

    # Also keep an 'origin' or region column if present (original 'tooth_class' in schema)
    if 'tooth_class' in df.columns and 'tooth_class' not in ['Anterior', 'Posterior', 'Premolar']:
        # if original 'tooth_class' column means something else (region), keep as 'origin'
        df['origin'] = df['tooth_class']

    # Standardize genus labels (strip whitespace, consistent capitalization)
    df['genus'] = df['genus'].str.strip()

    # Common expected labels in dataset: 'Homo sapiens', 'Pan', 'Pongo', 'Papio'
    # If some genus labels are abbreviated or different (e.g., 'Homo'), attempt simple normalization
    df['genus'] = df['genus'].replace({
        'Homo': 'Homo sapiens',
        'H. sapiens': 'Homo sapiens',
        'Homo_sapiens': 'Homo sapiens',
        'pan': 'Pan',
        'Pongo sp.': 'Pongo',
        'Papio sp.': 'Papio'
    })

    # Standardize tooth_class to a small set (Anterior, Posterior, Premolar) if possible
    if 'tooth_class' in df.columns:
        df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
        # unify common representations
        df['tooth_class'] = df['tooth_class'].replace({
            'Ant': 'Anterior', 'ant': 'Anterior', 'Post': 'Posterior', 'post': 'Posterior',
            'PM': 'Premolar', 'pm': 'Premolar'
        })
        # Where tooth_class is missing but origin contains tooth-type-like values, try to fill
        df.loc[df['tooth_class'].str.lower().isin(['anterior','posterior','premolar']) == False, 'tooth_class'] = df.loc[df['tooth_class'].str.lower().isin(['anterior','posterior','premolar']) == False, 'tooth_class']
    else:
        df['tooth_class'] = np.nan

    # Calculate proportion missing for binomial modeling; will be used as the endogenous variable in the GLM with weights
    df['prop_missing'] = np.nan
    has_valid_counts = (~df['num_missing'].isna()) & (~df['n_sockets'].isna())
    df.loc[has_valid_counts, 'num_missing'] = df.loc[has_valid_counts, 'num_missing'].astype(float)
    df.loc[has_valid_counts, 'n_sockets'] = df.loc[has_valid_counts, 'n_sockets'].astype(float)

    # Clip num_missing to be within [0, n_sockets]
    df.loc[has_valid_counts, 'num_missing'] = df.loc[has_valid_counts, 'num_missing'].clip(lower=0)
    # Where num_missing might exceed n_sockets due to data issues, cap it
    exceed_mask = has_valid_counts & (df['num_missing'] > df['n_sockets'])
    if exceed_mask.any():
        df.loc[exceed_mask, 'num_missing'] = df.loc[exceed_mask, 'n_sockets']

    valid_for_prop = has_valid_counts & (df['n_sockets'] > 0)
    df.loc[valid_for_prop, 'prop_missing'] = df.loc[valid_for_prop, 'num_missing'] / df.loc[valid_for_prop, 'n_sockets']

    # Drop rows that cannot be used in binomial model: missing genus, missing num_missing or n_sockets or n_sockets <= 0
    df = df[~df['genus'].isna()]
    df = df[valid_for_prop]

    # Keep only genera of interest (Homo sapiens, Pan, Pongo, Papio). If others are present, they will be dropped
    allowed_genera = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
    df = df[df['genus'].isin(allowed_genera)]

    # Coerce types for modeling
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['genus'] = df['genus'].astype('category')

    # Reset index
    df = df.reset_index(drop=True)

    # Final columns relevant for modeling
    final_cols = ['specimen', 'genus', 'tooth_class', 'num_missing', 'n_sockets', 'prop_missing', 'age_at_death', 'age_sd', 'sex_male_prob']
    # Ensure all final columns exist in df (add missing ones as NaN)
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression to model AMTL frequency (num_missing / n_sockets) with genus as the main predictor
    while controlling for age_at_death, sex_male_prob, and tooth_class.

    Modeling approach:
      - Use statsmodels' GLM with family=Binomial. Model prop_missing as the endogenous and supply n_sockets as weights.
      - Formula: prop_missing ~ C(genus) + age_at_death + sex_male_prob + C(tooth_class)
      - Return the fitted model object. Also print a concise summary and compute exponentiated coefficients (odds ratios) with 95% CIs.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    df = df.copy()

    # Ensure required columns exist
    required = ['prop_missing', 'n_sockets', 'genus', 'age_at_death', 'sex_male_prob', 'tooth_class', 'num_missing']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Drop rows with missing predictors (we already cleaned much in transform but double-check)
    model_df = df.dropna(subset=['prop_missing', 'n_sockets', 'genus', 'tooth_class', 'age_at_death', 'sex_male_prob'])

    # If there are very small denominators, remove or warn
    model_df = model_df[model_df['n_sockets'] > 0]

    # Fit GLM (binomial) with proportion response and weights = n_sockets
    # Using proportion as endog and weights allows modeling successes/trials in GLM
    formula = 'prop_missing ~ C(genus) + age_at_death + sex_male_prob + C(tooth_class)'
    glm_binom = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial(), weights=model_df['n_sockets'])
    res = glm_binom.fit()

    # Print summary and odds ratios
    print(res.summary())

    # Compute odds ratios and 95% CI
    params = res.params
    conf = res.conf_int()
    or_df = (np.exp(params).rename('odds_ratio')).to_frame()
    or_df['ci_lower'] = np.exp(conf[0])
    or_df['ci_upper'] = np.exp(conf[1])
    print('\nOdds ratios with 95% CI:')
    print(or_df)

    # Additional check for overdispersion: compare residual deviance to df_resid
    deviance = res.deviance
    df_resid = int(res.df_resid)
    dispersion = deviance / df_resid if df_resid > 0 else np.nan
    print(f"\nDeviance: {deviance:.2f}, df_resid: {df_resid}, dispersion (deviance/df_resid): {dispersion:.3f}")
    if dispersion > 1.5:
        print("Note: dispersion > 1.5 suggests potential overdispersion. Consider a quasi-binomial/GEE or add random effects.")

    # Return the fitted results object (so the caller can inspect coefficients, contrasts, etc.)
    return res


