from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/anonymize_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe. The function:
      - drops rows missing the primary IV (feature6) or DV (feature7),
      - converts and renames relevant columns,
      - encodes binary categorical controls as 0/1,
      - derives a log enrollment covariate and a squared beauty term,
      - drops rows with missing values in any model-required columns.
    Returns the transformed dataframe with the exact column names used in modeling.
    """
    df = df.copy()

    # Ensure the main columns exist and drop rows with missing IV or DV
    df = df.dropna(subset=['feature6', 'feature7'])

    # Numeric conversions and renaming
    df['beauty'] = pd.to_numeric(df['feature6'], errors='coerce')
    df['eval_score'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['age'] = pd.to_numeric(df['feature3'], errors='coerce')

    # Binary / factor conversions (map known string values to 0/1). Use .astype(str).str.lower() to be robust.
    df['minority'] = df['feature2'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['female'] = df['feature4'].astype(str).str.lower().map({'female': 1, 'male': 0})
    df['single_credit'] = df['feature5'].astype(str).str.lower().map({'single': 1, 'more': 0})
    df['upper_division'] = df['feature8'].astype(str).str.lower().map({'upper': 1, 'lower': 0})
    df['native_english'] = df['feature9'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['tenure_track'] = df['feature10'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Enrollment / counts / instructor id
    df['n_eval_participants'] = pd.to_numeric(df['feature11'], errors='coerce')
    df['enrollment'] = pd.to_numeric(df['feature12'], errors='coerce')
    df['instructor_id'] = pd.to_numeric(df['feature13'], errors='coerce')

    # Derived variables
    # log enrollment: clip to avoid log(0)
    df['log_enrollment'] = np.log(df['enrollment'].clip(lower=1))
    df['beauty_sq'] = df['beauty'] ** 2

    # Final required columns for modelling
    required = [
        'beauty', 'beauty_sq', 'eval_score', 'age', 'female', 'minority', 'tenure_track',
        'single_credit', 'upper_division', 'native_english', 'log_enrollment',
        'n_eval_participants', 'instructor_id'
    ]

    # Drop rows with missing data in any required column
    df = df.dropna(subset=required)

    # Keep only relevant columns (but retain original columns if downstream wants them)
    # Here we keep all original plus derived; if desired this can be restricted.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models to estimate the effect of instructor beauty on student evaluations.

    Two complementary specifications are returned:
      1) OLS with clustered standard errors at the instructor level (robust to within-instructor correlation),
      2) Linear mixed effects model with a random intercept for instructor (accounts for instructor-level unobserved heterogeneity).

    The formula includes a quadratic beauty term to allow nonlinearity and the set of controls defined in transform().
    Returns a dict with both fitted results objects: {'ols_clustered': ..., 'mixedlm': ...}
    """
    import statsmodels.formula.api as smf

    # Model formula: DV on beauty (linear + squared) plus controls
    formula = (
        'eval_score ~ beauty + beauty_sq + age + female + minority + tenure_track + '
        'single_credit + upper_division + native_english + log_enrollment + n_eval_participants'
    )

    # 1) OLS (clustered SEs by instructor)
    ols = smf.ols(formula, data=df).fit()
    # Clustered standard errors by instructor_id
    try:
        ols_clustered = ols.get_robustcov_results(cov_type='cluster', groups=df['instructor_id'])
    except Exception:
        # Fallback: if clustering fails, return the plain OLS fit
        ols_clustered = ols

    # 2) Linear mixed effects model with random intercept for instructor
    # This captures instructor-level unobserved heterogeneity as random effects.
    try:
        md = smf.mixedlm(formula, df, groups=df['instructor_id'])
        mixed = md.fit(reml=False)
    except Exception as e:
        mixed = None

    # Pack and return results
    results = {
        'ols_clustered': ols_clustered,
        'mixedlm': mixed
    }
    return results


