from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe expected by the model.

    Produces the following columns required for modeling:
      - MajorityChoice: binary outcome (1 if y==2, else 0)
      - age_c: age centered around the sample mean
      - age_c_sq: quadratic term of centered age
      - culture: categorical site identifier (category dtype)
      - is_boy: gender recoded to 1=boy, 0=girl (NaN if gender missing)
      - majority_first: ensured to be 0/1 integer (filled 0 if missing)

    The function drops rows missing the core variables (y, age, culture).
    """
    df = df.copy()

    # Drop rows missing core variables needed for analysis
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Create binary outcome: 1 if child chose the majority option (y == 2), else 0
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Ensure majority_first exists and is integer 0/1; if missing, fill with 0
    if 'majority_first' in df.columns:
        df['majority_first'] = df['majority_first'].fillna(0).astype(int)
    else:
        df['majority_first'] = 0

    # Recode gender to binary is_boy (1=boy where original gender==2, 0=girl where gender==1)
    if 'gender' in df.columns:
        # keep NaN if gender missing
        df['is_boy'] = np.where(df['gender'].isna(), np.nan, (df['gender'] == 2).astype(int))
    else:
        df['is_boy'] = np.nan

    # Convert culture to categorical for modeling; keep original integer codes but set dtype
    df['culture'] = df['culture'].astype('category')

    # Center age for interpretability and create quadratic term
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c_sq'] = df['age_c'] ** 2

    # Return dataframe containing at least the required columns
    required_cols = ['MajorityChoice', 'age_c', 'age_c_sq', 'culture', 'is_boy', 'majority_first']
    # It's fine if df contains more columns, but ensure required ones exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after transform: {missing}")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression to test how reliance on the majority develops with age
    and whether the age effect differs across cultural contexts.

    Model specification (fixed effects approach):
      MajorityChoice ~ age_c * C(culture) + age_c_sq + is_boy + majority_first

    - age_c * C(culture) fits separate age slopes for each culture (tests culture x age interaction).
    - age_c_sq captures non-linear age trends shared across sites.
    - is_boy and majority_first control for gender and demonstration order effects.

    Returns the fitted GLM results object (binomial family).
    """
    import statsmodels.formula.api as smf

    # Ensure culture is categorical for the formula. If already categorical, this is a no-op.
    df = df.copy()
    df['culture'] = df['culture'].astype('category')

    formula = 'MajorityChoice ~ age_c * C(culture) + age_c_sq + is_boy + majority_first'

    # Fit binomial GLM (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = model.fit()

    # Return the fitted results object for inspection (summary, coefficients, CI, etc.)
    return results


