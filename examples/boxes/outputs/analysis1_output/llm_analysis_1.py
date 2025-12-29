from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/boxes/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Produces the following columns used by the model:
      - MajorityChoice: binary outcome (1 if y==2, else 0)
      - age_c: age centered around the sample mean
      - is_boy: gender coded 0=girl, 1=boy
      - majority_first: ensures integer 0/1
      - culture: kept as-is (site id); used as categorical moderator in the model
    """
    # Work on a copy to avoid modifying the input in-place
    df = df.copy()

    # Drop rows missing any variables needed for the modeling
    required_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    df = df.dropna(subset=required_cols)

    # Dependent variable: majority choice (y == 2)
    # Original coding: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Independent variable: center age to aid interpretation of interactions
    df['age_c'] = df['age'] - df['age'].mean()

    # Control: gender -> binary indicator is_boy (0 = girl (1), 1 = boy (2))
    # Original coding: 1 = girl, 2 = boy
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Ensure culture is present and integer (site id)
    # If culture is not integer, try to coerce
    try:
        df['culture'] = df['culture'].astype(int)
    except Exception:
        # leave as-is if coercion fails
        pass

    # Keep only rows with valid binary outcome (MajorityChoice created above)
    df = df[df['MajorityChoice'].isin([0, 1])].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression testing whether the probability of choosing the majority option
    changes with age and whether that age effect differs across cultural sites.

    Model specification (fixed effects):
      MajorityChoice ~ age_c * C(culture) + is_boy + majority_first

    - age_c * C(culture) tests for different developmental slopes (age effects) across cultures.
    - is_boy and majority_first are included as covariates.

    The function returns the fitted model object with cluster-robust standard errors by culture
    (if clustering routine is available for the fitted object).
    """
    import statsmodels.formula.api as smf

    # Work on a copy to be safe
    df_model = df.copy()

    # Formula: interaction between centered age and culture (culture treated as categorical)
    formula = 'MajorityChoice ~ age_c * C(culture) + is_boy + majority_first'

    # Fit logistic regression (binomial logit)
    model_fit = smf.logit(formula=formula, data=df_model).fit(disp=False)

    # Obtain cluster-robust standard errors clustered by culture (site)
    # If clustering fails, return the original fitted model.
    try:
        results_cluster = model_fit.get_robustcov_results(cov_type='cluster', groups=df_model['culture'])
    except Exception:
        # Fallback: return the conventional fitted model
        results_cluster = model_fit

    return results_cluster


