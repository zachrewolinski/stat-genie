from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Produces the following new columns used in modeling:
    - Choice: integer copy of original y (1=unchosen, 2=majority, 3=minority)
    - demonstrated_choice: binary (1 if Choice in {2,3} i.e. child chose a demonstrated option; 0 if unchosen)
    - majority_choice: binary among demonstrated choices (1 if majority (2), 0 if minority (3); NaN if Choice==1)
    - age_z: standardized age (z-score)
    - culture_cat: categorical culture label as string (e.g., 'C1', 'C2', ...)
    - is_male: binary (1=boy, 0=girl)

    Drops rows missing any of the key variables used in these transforms (y, age, culture, gender, majority_first).
    """
    # Work on a copy to avoid modifying original
    df = df.copy()

    # Drop rows with missing critical fields
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Keep original choice coding but standardize the name for modeling
    df['Choice'] = df['y'].astype(int)

    # Binary: did the child pick one of the demonstrated options (majority or minority)?
    df['demonstrated_choice'] = df['Choice'].isin([2, 3]).astype(int)

    # For those who picked a demonstrated option, was it the majority (1) or minority (0)?
    # For nondemonstrated choices this will be NaN
    df['majority_choice'] = np.where(df['Choice'] == 2, 1,
                                     np.where(df['Choice'] == 3, 0, np.nan))

    # Standardize age (z-score) for interpretability and to stabilize interactions
    df['age_z'] = (df['age'] - df['age'].mean()) / df['age'].std()

    # Create a categorical label for culture (string) so formula interface treats it as a factor
    # Prepend 'C' to preserve categorical ordering as strings (e.g., 'C1')
    df['culture_cat'] = 'C' + df['culture'].astype(int).astype(str)

    # Gender: convert to is_male flag (1 = boy (originally coded 2), 0 = girl (1))
    df['is_male'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Final check: keep only rows with valid Choice codes (1,2,3)
    df = df[df['Choice'].isin([1, 2, 3])]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs statistical models to address (a) children's reliance on social information
    (demonstrated vs undemonstrated choices) and (b) children's preference for
    the majority when they use social information (majority vs minority).

    Two logistic regression models are fit using statsmodels' formula API:
    1) demonstrated_choice ~ age_z + C(culture_cat) + is_male + majority_first + age_z:C(culture_cat)
       (binary logistic regression predicting whether a child chose a demonstrated option)
    2) majority_choice ~ age_z + C(culture_cat) + is_male + majority_first + age_z:C(culture_cat)
       (binary logistic regression among subset who chose a demonstrated option)

    Both models include an age-by-culture interaction to test whether developmental
    trajectories differ across cultures.

    Returns a dict with fitted model results (statsmodels objects):
      {'demonstrated_model': <BinaryResults>, 'majority_model': <BinaryResults>, 'n_demo': int}
    """
    import statsmodels.formula.api as smf
    import warnings

    results = {}

    # Model 1: reliance on social information (demonstrated vs undemonstrated)
    formula1 = 'demonstrated_choice ~ age_z + C(culture_cat) + is_male + majority_first + age_z:C(culture_cat)'
    try:
        m1 = smf.logit(formula=formula1, data=df).fit(disp=False)
    except Exception as e:
        # If convergence or perfect separation occurs, raise informative error
        raise RuntimeError(f"Model 1 failed to fit: {e}")

    results['demonstrated_model'] = m1

    # Model 2: preference for majority vs minority among those who used social information
    df_demo = df[df['demonstrated_choice'] == 1].copy()
    n_demo = df_demo.shape[0]
    results['n_demo'] = int(n_demo)

    if n_demo < 30:
        warnings.warn(
            f"Small sample for majority-vs-minority model (n={n_demo}). Coefficients may be unstable.")

    # Drop rows where majority_choice is missing (should only be missing when Choice==1)
    df_demo = df_demo.dropna(subset=['majority_choice'])

    if df_demo['majority_choice'].nunique() < 2:
        warnings.warn('No variation in majority_choice in the demonstrated subset; model cannot be fit.')
        results['majority_model'] = None
        return results

    formula2 = 'majority_choice ~ age_z + C(culture_cat) + is_male + majority_first + age_z:C(culture_cat)'
    try:
        m2 = smf.logit(formula=formula2, data=df_demo).fit(disp=False)
    except Exception as e:
        raise RuntimeError(f"Model 2 failed to fit: {e}")

    results['majority_model'] = m2

    # Optionally: print summaries for immediate inspection (commented out by default)
    # print(m1.summary())
    # print(m2.summary())

    return results


