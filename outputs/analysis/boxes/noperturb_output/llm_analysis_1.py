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
    Transform the raw dataset into a dataframe ready for multinomial modeling.

    Produces the following added/modified columns (all used by the model):
      - y_mn: integer 0/1/2 for multinomial outcome (0=unchosen,1=majority,2=minority)
      - choice_label: human-readable label of the choice (unchosen/majority/minority)
      - age_z: standardized age (z-score)
      - is_boy: binary encoding of gender (1 = boy, 0 = girl)
      - majority_first: ensured to be integer 0/1
      - culture: kept as the original culture id (converted to string for categorical handling downstream)

    Rows with missing values in any of the required modeling columns are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Required columns
    required_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    # Drop rows missing any of these
    df = df.dropna(subset=required_cols)

    # Ensure types
    # y should be integers 1,2,3 per schema. Convert to int and check range
    df['y'] = df['y'].astype(int)

    # Create human-readable label (optional but useful)
    label_map = {1: 'unchosen', 2: 'majority', 3: 'minority'}
    df['choice_label'] = df['y'].map(label_map)

    # Create y_mn for statsmodels MNLogit (0..J-1)
    # Map 1->0, 2->1, 3->2
    df['y_mn'] = df['y'] - 1

    # Standardize age (z-score)
    df['age_z'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)

    # Encode gender as binary is_boy (schema: 1=girl, 2=boy)
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Keep culture as a categorical-like column; convert to string so that get_dummies in modeling treats each id distinctly
    # (this avoids accidental numeric interpretation of culture ids)
    df['culture'] = df['culture'].astype(int).astype(str)

    # Final check: drop any rows with missing values in the newly created columns
    model_cols = ['y_mn', 'age_z', 'is_boy', 'majority_first', 'culture']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a multinomial logistic regression (MNLogit) predicting choice (y_mn)
    from age, culture, and controls (is_boy, majority_first). The model
    includes main effects of age (age_z) and culture (one-hot dummies with
    the first culture as reference) and interaction terms between age_z and
    each culture dummy to test whether developmental trajectories differ by culture.

    Returns the fitted statsmodels MNLogit result object.
    """
    import statsmodels.api as sm

    # Work on a copy
    data = df.copy()

    # Endogenous variable: y_mn (0,1,2). Ensure integer dtype
    endog = data['y_mn'].astype(int)

    # Base exogenous predictors
    exog = pd.DataFrame({'intercept': 1.0, 'age_z': data['age_z'], 'is_boy': data['is_boy'], 'majority_first': data['majority_first']})

    # Create culture dummies (drop first to avoid multicollinearity)
    culture_dummies = pd.get_dummies(data['culture'], prefix='culture', drop_first=True)
    # Add culture dummies to exog
    exog = pd.concat([exog, culture_dummies], axis=1)

    # Add interaction terms: age_z x each culture dummy
    for col in culture_dummies.columns:
        inter_name = f'age_z_x_{col}'
        exog[inter_name] = exog['age_z'] * exog[col]

    # Ensure no NaNs and proper dtype
    exog = exog.astype(float)

    # Fit multinomial logit using statsmodels MNLogit
    # statsmodels' MNLogit models the log-odds of each non-reference category vs reference.
    # Here the reference category will be the first integer value (0: 'unchosen').
    model = sm.MNLogit(endog, exog)

    # fit; increase maxiter if necessary
    try:
        result = model.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        # fallback to default solver with more iterations
        result = model.fit(method='bfgs', maxiter=500, disp=False)

    # You may want to inspect result.summary() in an interactive session.
    # Return the full result object so the caller can examine params, p-values, predictions, etc.
    return result


