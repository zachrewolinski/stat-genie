from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables required for modeling instructor beauty -> eval.

    Produces the following columns used in the model:
      - eval: dependent variable (keeps original)
      - beauty_z: standardized beauty score (mean 0, sd 1)
      - beauty_z_sq: squared standardized beauty (to test nonlinearity)
      - age: numeric age
      - gender_male: 1 if gender == 'male', 0 otherwise
      - minority_yes: 1 if minority == 'yes', 0 otherwise
      - tenure_yes: 1 if tenure == 'yes', 0 otherwise
      - native_yes: 1 if native == 'yes', 0 otherwise
      - single_credit: 1 if credits == 'single', 0 otherwise
      - division_upper: 1 if division == 'upper', 0 otherwise
      - log_students: log(students + 1)
      - prof: integer professor identifier (used for clustering / random effects)
    """
    df = df.copy()

    # Keep rows with required numeric outcome and beauty
    df = df.dropna(subset=['eval', 'beauty']).reset_index(drop=True)

    # Standardize beauty for interpretability
    # use population standard deviation (ddof=0) to get standardized z
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if beauty_std == 0 or np.isnan(beauty_std):
        df['beauty_z'] = 0.0
    else:
        df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std
    df['beauty_z_sq'] = df['beauty_z'] ** 2

    # Ensure students numeric and fill missing with median
    if 'students' in df.columns:
        df['students'] = pd.to_numeric(df['students'], errors='coerce')
        median_students = df['students'].median()
        df['students'] = df['students'].fillna(median_students)
    else:
        df['students'] = 0
    df['log_students'] = np.log(df['students'] + 1)

    # Clean and encode categorical covariates into binary columns
    def clean_str_col(col):
        return df[col].astype(str).str.strip().str.lower() if col in df.columns else pd.Series([''] * len(df))

    df['gender_male'] = (clean_str_col('gender') == 'male').astype(int)
    df['minority_yes'] = (clean_str_col('minority') == 'yes').astype(int)
    df['tenure_yes'] = (clean_str_col('tenure') == 'yes').astype(int)
    df['native_yes'] = (clean_str_col('native') == 'yes').astype(int)
    df['single_credit'] = (clean_str_col('credits') == 'single').astype(int)
    df['division_upper'] = (clean_str_col('division') == 'upper').astype(int)

    # Age numeric handling
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['age'] = np.nan
    # Drop rows lacking age (age is an important control); alternatively one could impute.
    df = df.dropna(subset=['age']).reset_index(drop=True)

    # Ensure professor id exists and is integer for clustering / mixed models
    if 'prof' in df.columns:
        df['prof'] = pd.to_numeric(df['prof'], errors='coerce')
        df = df.dropna(subset=['prof']).reset_index(drop=True)
        df['prof'] = df['prof'].astype(int)
    else:
        # if no professor identifier, create a single group (will reduce ability to cluster)
        df['prof'] = 0

    # Final column selection (keeps eval and the transforms above)
    final_cols = ['eval', 'beauty_z', 'beauty_z_sq', 'age', 'gender_male', 'minority_yes', 'tenure_yes', 'native_yes', 'single_credit', 'division_upper', 'log_students', 'prof']
    # Ensure all final columns are present (if any missing, initialize to default)
    for c in final_cols:
        if c not in df.columns:
            if c == 'eval':
                df[c] = np.nan
            elif c == 'prof':
                df[c] = 0
            else:
                df[c] = 0

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit models to estimate the association between instructor beauty and student evaluations.

    Two complementary specifications are estimated:
      1) OLS with covariates and professor-clustered standard errors.
      2) Linear mixed-effects model with a random intercept for professor (accounts for within-professor correlation).

    Returns a dictionary with the fitted models and their summaries.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure dataframe contains the required columns
    required = ['eval', 'beauty_z', 'beauty_z_sq', 'age', 'gender_male', 'minority_yes', 'tenure_yes', 'native_yes', 'single_credit', 'division_upper', 'log_students', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Drop any remaining rows with missing values in model variables
    model_df = df.dropna(subset=required).copy()

    # Formula
    formula = 'eval ~ beauty_z + beauty_z_sq + age + gender_male + minority_yes + tenure_yes + native_yes + single_credit + division_upper + log_students'

    # 1) OLS with clustering by professor
    ols_res = smf.ols(formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['prof']})

    # 2) Linear mixed-effects model with random intercept for professor
    # Use statsmodels MixedLM; note that if 'prof' has many unique values relative to data, this can still be fitted.
    try:
        md = sm.MixedLM.from_formula(formula, groups=model_df['prof'], data=model_df)
        mdf = md.fit(reml=False)
    except Exception as e:
        # If mixed model fails to converge, return the error in place of the fitted model
        mdf = {'error': str(e)}

    # Return both fit objects and summaries for downstream inspection
    results = {
        'ols_result': ols_res,
        'ols_summary': ols_res.summary().as_text(),
        'mixedlm_result': mdf,
        'n_obs': int(model_df.shape[0]),
        'n_professors': int(model_df['prof'].nunique())
    }
    return results


