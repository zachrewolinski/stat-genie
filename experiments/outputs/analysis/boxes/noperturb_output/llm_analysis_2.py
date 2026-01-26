from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe to produce the columns required for modeling.

    Outputs (columns guaranteed to exist for modeling):
      - y: outcome (1=unchosen,2=majority,3=minority) (kept as provided)
      - age: original age in years (kept)
      - age_c: age centered at the sample mean (numeric)
      - age_group: categorical age bins (strings): '4-6','7-9','10-12','13-14'
      - culture: original culture/site id (kept, converted to categorical dtype)
      - is_male: recoded gender where 1=boy (original gender: 1=girl, 2=boy)
      - majority_first: kept as-is (0/1)

    The function drops rows with missing values on the necessary columns.
    """
    import numpy as np
    import pandas as pd

    # Work on a copy to avoid mutating the input dataframe in-place
    df = df.copy()

    # Required input columns (based on provided schema)
    required_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Ensure correct dtypes
    # y should be integer categories 1,2,3
    df['y'] = df['y'].astype(int)

    # gender: original coding 1=girl, 2=boy. Create is_male: 1 if boy else 0.
    df['is_male'] = df['gender'].apply(lambda g: 1 if int(g) == 2 else 0)

    # age: keep original, but create centered version for modeling stability
    df['age'] = df['age'].astype(float)
    df['age_c'] = df['age'] - df['age'].mean()

    # Create age bins (developmental stages) for description/stratified summaries
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]  # cut points giving 4-6,7-9,10-12,13-14
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)

    # Culture: keep as provided but make categorical for modeling
    # Some toolchains treat numeric categories as continuous; keep dtype categorical so formula-based design matrices treat it as factor
    df['culture'] = df['culture'].astype('category')

    # majority_first should be binary 0/1; enforce integer dtype
    df['majority_first'] = df['majority_first'].astype(int)

    # Optionally, drop rows with categories outside expected ranges (defensive)
    df = df[df['y'].isin([1, 2, 3])]

    # Reset index before returning
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a multinomial logistic regression (multinomial logit) predicting the 3-way outcome `y` from:
      - main effects: centered age (age_c) and culture (categorical)
      - interaction: age_c * culture  (tests whether developmental change differs across cultures)
      - controls: is_male and majority_first

    Returns the fitted statsmodels MNLogit results object.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from patsy import dmatrix

    # Ensure required columns exist
    required = ['y', 'age_c', 'culture', 'is_male', 'majority_first']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Endogenous variable: integer-coded categories 1,2,3
    endog = df['y'].astype(int)

    # Design matrix (exogenous). Use patsy to create dummy variables for culture and the interaction.
    # The formula below creates an intercept automatically.
    exog = dmatrix('age_c * C(culture) + is_male + majority_first', df, return_type='dataframe')

    # Convert to plain numpy arrays for statsmodels MNLogit
    # MNLogit expects exog shape (nobs, k) and endog as a 1d array of integers labeling the choice
    X = np.asarray(exog)
    y = np.asarray(endog)

    # Fit the multinomial logistic regression using statsmodels MNLogit
    # Note: MNLogit treats the lowest integer label in endog as the reference category by default.
    model_mn = sm.MNLogit(y, X)

    # Use a robust fitting procedure; suppress iteration output
    try:
        fit = model_mn.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        # Fallback to default settings if newton fails
        fit = model_mn.fit(disp=False)

    # Return the fitted result object (user can call .summary(), .params, .predict(), etc.)
    return fit


