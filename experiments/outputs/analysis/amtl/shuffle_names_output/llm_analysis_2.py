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
    Transform the raw dataframe into a cleaned dataframe with correctly mapped columns for AMTL analysis.

    Important note on column mapping: the provided dataset schema has shifted/descriptive labels that do not match the logical contents.
    Empirically the following mapping is applied (guided by dataset_description):
      - stdev_age  -> number of missing teeth for the given tooth class (n_missing)
      - prob_male  -> number of observable sockets (n_sockets)
      - num_amtl   -> estimated age at death (age_est)
      - sockets    -> assigned uncertainty / sd of age-at-death (age_sd)
      - pop        -> sex estimate (probability male) (sex_prob_male)
      - age        -> specimen genus (Pan, Pongo, Papio, Homo sapiens) (genus)
      - genus      -> tooth class label (Anterior, Posterior, Premolar) (tooth_class)

    The function performs type conversions, cleans/counts rounding, computes proportions, and creates an is_human indicator.
    """
    df = df.copy()

    # Map columns based on the schema's shifted descriptions
    # Use the original column names but create logically-named new columns used in the model.
    # If any of the source columns are missing, this will raise a KeyError so users can inspect the dataset.
    # Convert to numeric where appropriate; coerce errors to NaN for later dropping.
    df['n_missing'] = pd.to_numeric(df['stdev_age'], errors='coerce')
    df['n_sockets'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df['age_est'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['age_sd'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['sex_prob_male'] = pd.to_numeric(df['pop'], errors='coerce')

    # Genus (taxon) is provided in the column 'age' per the schema mapping
    df['genus'] = df['age'].astype(str)

    # Tooth class is actually in the column named 'genus' in the schema
    df['tooth_class'] = df['genus'].astype(str)

    # Standardize tooth_class labels (common expected values: Anterior, Posterior, Premolar)
    df['tooth_class'] = df['tooth_class'].str.strip().str.title()

    # Round sockets and n_missing to integers (sockets must be positive integer counts)
    # Many source values appear as floats; round them sensibly. Keep nullable integer dtype to preserve NaNs.
    df['n_sockets'] = df['n_sockets'].round().astype('Int64')
    df['n_missing'] = df['n_missing'].round().astype('Int64')

    # Ensure logical bounds: 0 <= n_missing <= n_sockets. If n_sockets is missing or <= 0, mark row for removal.
    # Where n_missing is negative or greater than sockets, clip to feasible range.
    valid_sockets_mask = df['n_sockets'].notna() & (df['n_sockets'] > 0)
    df = df[valid_sockets_mask].copy()

    # Clip n_missing between 0 and n_sockets; if n_missing is missing, set to 0 (conservative) or drop? We'll drop rows with missing n_missing.
    df = df[df['n_missing'].notna()].copy()
    df['n_missing'] = df.apply(lambda r: int(max(0, min(int(r['n_missing']), int(r['n_sockets'])))), axis=1)
    df['n_sockets'] = df['n_sockets'].astype(int)

    # Proportion of AMTL
    df['amtl_prop'] = df['n_missing'] / df['n_sockets']

    # Create is_human indicator from genus values. Accept common Homo labels such as 'Homo', 'Homo sapiens', 'Homo_sapiens'
    df['genus'] = df['genus'].str.strip()
    df['is_human'] = df['genus'].str.contains('Homo', case=False, na=False).astype(int)

    # Some basic cleaning for age_est and sex_prob_male
    df['age_est'] = pd.to_numeric(df['age_est'], errors='coerce')
    df['sex_prob_male'] = pd.to_numeric(df['sex_prob_male'], errors='coerce')

    # Drop rows missing critical model columns
    required_cols = ['specimen', 'n_missing', 'n_sockets', 'amtl_prop', 'is_human', 'tooth_class']
    df = df.dropna(subset=required_cols)

    # Reset index and return only the columns needed downstream (plus helpful metadata)
    out_cols = [
        'specimen',
        'n_missing',
        'n_sockets',
        'amtl_prop',
        'is_human',
        'genus',
        'tooth_class',
        'age_est',
        'age_sd',
        'sex_prob_male'
    ]

    # Ensure all requested out_cols are present in df; if not, add them as NA to preserve shape
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) regression for AMTL counts with genus (human vs non-human) as main predictor,
    controlling for age, sex, and tooth class. Clustered standard errors by specimen are used to account for
    potential non-independence within specimen rows.

    Returns a dictionary containing the fitted model object, exponentiated coefficients (odds ratios),
    95% confidence intervals for odds ratios, and the textual summary.
    """
    df = df.copy()

    # Required modeling columns
    required = ['n_missing', 'n_sockets', 'is_human', 'age_est', 'sex_prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build response as proportion with frequency weights = trials (n_sockets)
    endog = df['n_missing'] / df['n_sockets']

    # Build design matrix: is_human, age_est (centered), sex_prob_male, and tooth_class dummies
    exog = pd.DataFrame()
    exog['is_human'] = df['is_human'].astype(float)

    # Center age_est to improve interpretability (if available)
    if df['age_est'].notna().any():
        exog['age_est_c'] = df['age_est'].astype(float) - df['age_est'].astype(float).mean()
    else:
        exog['age_est_c'] = 0.0

    # sex probability (numeric 0-1) - fill missing with column mean for stability (flagging missing would be alternative)
    if df['sex_prob_male'].notna().any():
        exog['sex_prob_male'] = df['sex_prob_male'].astype(float).fillna(df['sex_prob_male'].mean())
    else:
        exog['sex_prob_male'] = 0.0

    # Tooth class dummies (drop first to avoid multicollinearity)
    tooth_dummies = pd.get_dummies(df['tooth_class'].astype(str), prefix='tooth', drop_first=True)
    exog = pd.concat([exog, tooth_dummies], axis=1)

    # Add constant
    exog = sm.add_constant(exog, has_constant='add')

    # Fit GLM Binomial using proportion endog and frequency weights = number of sockets
    # Use clustered standard errors by specimen to account for multiple rows per specimen
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial(), freq_weights=df['n_sockets'])
    try:
        res = glm_binom.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
    except Exception:
        # Fallback to default fit if clustering fails for any reason
        res = glm_binom.fit()

    # Exponentiate coefficients to get odds ratios and transform conf_int
    params = res.params
    conf = res.conf_int()
    or_vals = np.exp(params)
    conf_or = np.exp(conf)

    # Package results
    results = {
        'model': res,
        'odds_ratios': or_vals,
        'conf_int_or': conf_or,
        'summary': res.summary().as_text()
    }

    return results


