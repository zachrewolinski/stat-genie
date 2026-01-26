from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/add_features_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the final dataframe used in modeling.

    Adds the following columns (exact names used by the model):
      - trials: integer number of observable sockets (same as 'sockets')
      - amtl_prop: num_amtl / sockets (proportion missing)
      - is_human: binary indicator 1 if genus == 'Homo sapiens', else 0
      - age_centered: age minus mean(age)

    Also imputes missing prob_male and stdev_age with their medians (to retain rows) and ensures sockets > 0.
    """
    df = df.copy()

    # Keep rows that have the essential fields for binomial modeling
    required = ['num_amtl', 'sockets', 'genus', 'age']
    df = df.dropna(subset=required)

    # Ensure sockets is positive integer and convert to int
    df = df[df['sockets'] > 0].copy()
    df['sockets'] = df['sockets'].astype(int)

    # Trials for binomial model
    df['trials'] = df['sockets']

    # Proportion missing (outcome for convenience). Keep raw counts for binomial
    df['amtl_prop'] = df['num_amtl'] / df['trials']

    # Create binary human indicator. Handle whitespace and capitalization robustly
    df['genus'] = df['genus'].astype(str).str.strip()
    df['is_human'] = (df['genus'].str.lower() == 'homo sapiens').astype(int)

    # Center age for interpretability
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    mean_age = df['age'].mean()
    df['age_centered'] = df['age'] - mean_age

    # Impute prob_male and stdev_age with medians if missing (prefer retaining samples)
    if 'prob_male' in df.columns:
        median_prob_male = df['prob_male'].median()
        df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce').fillna(median_prob_male)
    else:
        # if missing entirely, add a neutral 0.5 column
        df['prob_male'] = 0.5

    if 'stdev_age' in df.columns:
        median_stdev_age = df['stdev_age'].median()
        df['stdev_age'] = pd.to_numeric(df['stdev_age'], errors='coerce').fillna(median_stdev_age)
    else:
        df['stdev_age'] = df['age_centered'].abs().median()

    # Ensure tooth_class exists and is a categorical variable
    if 'tooth_class' in df.columns:
        df['tooth_class'] = df['tooth_class'].astype(str).str.strip().astype('category')
    else:
        # If missing, create a single-category placeholder (will be dropped in practice)
        df['tooth_class'] = 'Unknown'
        df['tooth_class'] = df['tooth_class'].astype('category')

    # Ensure specimen column present for clustering; if missing, create unique ids per row
    if 'specimen' not in df.columns or df['specimen'].isnull().all():
        df['specimen'] = ['row_%d' % i for i in range(len(df))]
    else:
        df['specimen'] = df['specimen'].astype(str)

    # Final basic sanity checks: num_amtl between 0 and trials
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce').fillna(0).astype(int)
    df.loc[df['num_amtl'] < 0, 'num_amtl'] = 0
    df.loc[df['num_amtl'] > df['trials'], 'num_amtl'] = df.loc[df['num_amtl'] > df['trials'], 'trials']

    # Return only columns necessary for modeling plus originals for traceability
    cols_to_keep = [
        'specimen', 'genus', 'is_human',
        'num_amtl', 'sockets', 'trials', 'amtl_prop',
        'age', 'age_centered', 'stdev_age',
        'prob_male', 'tooth_class'
    ]
    # keep any that actually exist in df
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    return df[cols_to_keep].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logit) regression to test whether modern humans ('is_human') have higher AMTL
    after accounting for age, sex (prob_male), stdev_age, and tooth_class. Clustered standard
    errors by specimen are used to account for repeated measures per specimen (different tooth classes).

    Returns the statsmodels results object with clustered robust covariance.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Expect the dataframe to already be transformed by transform()
    required = ['num_amtl', 'trials', 'is_human', 'age_centered', 'prob_male', 'stdev_age', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('The following required columns are missing from the dataframe: %s' % missing)

    # Formula: model proportion with binomial family using trials as weights
    # Use categorical tooth_class via C(tooth_class).
    formula = 'amtl_prop ~ is_human + age_centered + prob_male + stdev_age + C(tooth_class)'

    # Fit GLM with Binomial family using weights=trials (number of trials per observation)
    # Use the proportion (amtl_prop) as endog with weights=trials to fit a binomial logistic regression.
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['trials'])
    res = model_glm.fit()

    # Obtain robust (clustered) standard errors clustered by specimen
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # Fallback: if clustered covariance fails, return the original fit and warn
        print('Warning: clustered robust covariance by specimen failed; returning non-clustered results.')
        res_cluster = res

    # Print summary (user can inspect) and return results object with clustered cov if available
    print(res_cluster.summary())
    return res_cluster


