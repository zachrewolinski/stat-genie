from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Note: the file read here is incidental; transform() should operate on any DataFrame passed to it.
# Keeping for compatibility with original file structure, but not used by the functions directly.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/shuffle_names_output/amtl.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw input dataframe to the analysis-ready dataframe.

    Produces final columns required by the modeling pipeline:
      ['specimen', 'genus', 'is_human', 'tooth_class', 'n_missing', 'n_sockets',
       'amtl_prop', 'age_at_death', 'age_se', 'male_prob']

    Ensures numeric columns use numpy dtypes (int64/float64) so that downstream
    modeling libraries (patsy/statsmodels) do not encounter pandas extension dtypes.
    """
    df = df.copy()

    # Map / derive columns (use original column names as present in the dataset)
    # n_missing: number of antemortem missing teeth for the given class
    df['n_missing'] = pd.to_numeric(df.get('stdev_age'), errors='coerce')
    # n_sockets: number of observable sockets (trials)
    df['n_sockets'] = pd.to_numeric(df.get('prob_male'), errors='coerce')
    # age at death (years)
    df['age_at_death'] = pd.to_numeric(df.get('num_amtl'), errors='coerce')
    # uncertainty of age-at-death
    df['age_se'] = pd.to_numeric(df.get('sockets'), errors='coerce')
    # sex estimate (probability male 0-1)
    df['male_prob'] = pd.to_numeric(df.get('pop'), errors='coerce')

    # genus (specimen taxon): original column named 'age' per schema
    # convert to string and strip whitespace; keep as object dtype
    df['genus'] = df.get('age').astype(str).str.strip()
    # tooth class: original column named 'genus' per schema (contains Anterior/Posterior/Premolar)
    df['tooth_class'] = df.get('genus').astype(str).str.strip()

    # specimen identifier (for possible clustering)
    if 'specimen' in df.columns:
        df['specimen'] = df['specimen'].astype(str)
    else:
        # if specimen id missing, create an index-based id (less preferred)
        df['specimen'] = df.index.astype(str)

    # Round n_missing and n_sockets to nearest integer where appropriate (keep as floats for now)
    df['n_missing'] = df['n_missing'].round()
    df['n_sockets'] = df['n_sockets'].round()

    # Ensure numeric coercion for controls as floats (may contain NaN)
    df['age_at_death'] = pd.to_numeric(df['age_at_death'], errors='coerce')
    df['age_se'] = pd.to_numeric(df['age_se'], errors='coerce')
    df['male_prob'] = pd.to_numeric(df['male_prob'], errors='coerce')

    # Cap/clean n_missing relative to n_sockets; use a safe row-wise operation
    def cap_missing(row):
        nm = row['n_missing']
        ns = row['n_sockets']
        try:
            if pd.isna(nm) or pd.isna(ns):
                return np.nan
            nm_i = int(nm)
            ns_i = int(ns)
        except Exception:
            return np.nan
        nm_i = max(nm_i, 0)
        if nm_i > ns_i:
            nm_i = ns_i
        return nm_i

    df['n_missing_capped'] = df.apply(cap_missing, axis=1)

    # Replace the original n_missing with capped values
    df['n_missing'] = df['n_missing_capped']
    df.drop(columns=['n_missing_capped'], inplace=True)

    # Drop rows that do not have the essential columns for a binomial model
    essential = ['n_missing', 'n_sockets', 'genus', 'tooth_class', 'age_at_death', 'male_prob']
    df = df.dropna(subset=essential)

    # Ensure n_sockets > 0
    # Convert n_sockets to numeric again (float) then to integer below
    df['n_sockets'] = pd.to_numeric(df['n_sockets'], errors='coerce')
    df = df[df['n_sockets'] > 0]

    # At this point, we've dropped rows missing essential values; safe to cast to concrete numpy dtypes.
    # Convert counts to integer numpy dtype
    df['n_missing'] = df['n_missing'].astype(int)
    df['n_sockets'] = df['n_sockets'].astype(int)

    # Proportion for modelling (used with binomial weights)
    df['amtl_prop'] = df['n_missing'] / df['n_sockets']
    df['amtl_prop'] = df['amtl_prop'].astype(float)

    # Binary indicator: modern human vs non-human primate
    # Ensure genus is string-like and lowercased for matching
    df['genus'] = df['genus'].astype(str).str.strip()
    df['is_human'] = df['genus'].str.lower().str.contains('homo', na=False).astype(int)

    # Treat tooth_class as a categorical variable with limited levels
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Ensure male_prob is numeric and constrained to [0,1] and cast to numpy float64
    df['male_prob'] = pd.to_numeric(df['male_prob'], errors='coerce').astype(float)
    df.loc[df['male_prob'] < 0, 'male_prob'] = 0.0
    df.loc[df['male_prob'] > 1, 'male_prob'] = 1.0

    # Ensure age_at_death and age_se are float64
    df['age_at_death'] = pd.to_numeric(df['age_at_death'], errors='coerce').astype(float)
    df['age_se'] = pd.to_numeric(df['age_se'], errors='coerce').astype(float)

    # Ensure specimen is object (string)
    df['specimen'] = df['specimen'].astype(str)

    # Final: keep only the columns needed for modeling
    keep_cols = ['specimen', 'genus', 'is_human', 'tooth_class', 'n_missing', 'n_sockets', 'amtl_prop', 'age_at_death', 'age_se', 'male_prob']
    df_out = df[keep_cols].reset_index(drop=True)

    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logit) GLM comparing AMTL in Homo sapiens versus non-human primates,
    controlling for age, sex, and tooth class. Uses the transformed dataframe produced by transform().

    Returns a dict containing primary and secondary model results and the model data.
    """
    # Copy to avoid modifying input
    data = df.copy()

    # Drop rows with missing essential modeling columns
    model_cols = ['n_missing', 'n_sockets', 'amtl_prop', 'is_human', 'genus', 'tooth_class', 'age_at_death', 'male_prob', 'age_se', 'specimen']
    data = data.dropna(subset=model_cols)

    # Ensure integer trials and proper dtypes (convert to numpy dtypes to avoid pandas extension dtypes)
    data['n_sockets'] = data['n_sockets'].astype(int)
    data['n_missing'] = data['n_missing'].astype(int)
    data['amtl_prop'] = data['amtl_prop'].astype(float)
    data['age_at_death'] = data['age_at_death'].astype(float)
    data['age_se'] = data['age_se'].astype(float)
    data['male_prob'] = data['male_prob'].astype(float)
    data['is_human'] = data['is_human'].astype(int)
    data['specimen'] = data['specimen'].astype(str)
    # Ensure tooth_class is categorical (patsy accepts pandas.Categorical)
    data['tooth_class'] = data['tooth_class'].astype('category')
    data['genus'] = data['genus'].astype(str)

    # Ensure integer trials > 0
    data = data[data['n_sockets'] > 0]

    # Create formula: primary model uses binary is_human indicator
    formula = 'amtl_prop ~ is_human + age_at_death + male_prob + C(tooth_class) + age_se'

    # Prepare freq_weights as a numpy array of floats to avoid pandas extension dtypes issues
    freq_weights = data['n_sockets'].to_numpy(dtype=float)

    # Fit GLM with binomial family using n_sockets as frequency weights so that the model reflects counts
    model_glm = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial(), freq_weights=freq_weights)
    res = model_glm.fit()

    # Obtain clustered robust standard errors by specimen (if specimen is available)
    try:
        res_clustered = res.get_robustcov_results(cov_type='cluster', groups=data['specimen'])
    except Exception:
        # If clustering fails for any reason (e.g., only one observation per cluster), fall back to default results
        res_clustered = res

    # Print a concise summary for the user
    print('Primary model (GLM Binomial with freq_weights = n_sockets)')
    print(res_clustered.summary())

    # Secondary model: genus as a categorical predictor (to inspect pairwise genus differences)
    formula2 = 'amtl_prop ~ C(genus) + age_at_death + male_prob + C(tooth_class) + age_se'
    model_glm2 = sm.GLM.from_formula(formula2, data=data, family=sm.families.Binomial(), freq_weights=freq_weights)
    res2 = model_glm2.fit()
    try:
        res2_clustered = res2.get_robustcov_results(cov_type='cluster', groups=data['specimen'])
    except Exception:
        res2_clustered = res2

    print('\nSecondary model (categorical genus)')
    print(res2_clustered.summary())

    # Return both fitted objects (primary and secondary) and their clustered-robust counterparts
    return {
        'primary_result': res,
        'primary_result_clustered': res_clustered,
        'genus_result': res2,
        'genus_result_clustered': res2_clustered,
        'model_data': data
    }