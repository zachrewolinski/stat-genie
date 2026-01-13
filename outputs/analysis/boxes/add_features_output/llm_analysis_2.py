from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataset to the analysis dataframe.

    Produces the following columns used in the model:
      - MajorityChoice: binary (1 = chose majority option, 0 = otherwise)
      - Age_centered: age (years) mean-centered
      - culture: categorical/site indicator as string (e.g., 'c1', 'c2', ...)
      - Female: binary female indicator (1 = girl, 0 = boy/other)
      - majority_first: coerced to integer 0/1 indicating whether majority was shown first

    Drops rows with missing values in variables required for the model.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe")

    # Drop rows with missing outcome or key predictors
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Dependent variable: majority choice (y == 2 indicates majority)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Create Female indicator (original coding: 1=girl, 2=boy)
    df['Female'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is 0/1 integer (if missing, keep as NaN so earlier dropna would have removed)
    # Some datasets encode it as booleans or ints; coerce safely
    df['majority_first'] = df['majority_first'].astype(float).astype('Int64')
    # If there are NA after coercion, leave them (already dropped if necessary)
    # Convert to plain int 0/1 where possible
    df['majority_first'] = df['majority_first'].astype(float).fillna(0).astype(int)

    # Convert culture to categorical string labels (prefix 'c' to avoid numeric interpretation)
    # Ensure culture is integer-like first
    df['culture'] = df['culture'].astype(int).astype(str)
    df['culture'] = 'c' + df['culture']

    # Center age to improve interpretability and numerical stability
    df['Age_centered'] = df['age'] - df['age'].mean()

    # Keep only columns necessary for modeling (but return full df to preserve other info)
    # Required columns for model: ['MajorityChoice', 'Age_centered', 'culture', 'Female', 'majority_first']
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial logistic regression testing how reliance on majority choices develops with age
    across cultural contexts. We include an Age x Culture interaction to allow developmental
    trajectories to vary by culture, and control for child gender and order (majority_first).

    Model: MajorityChoice ~ Age_centered * C(culture) + Female + majority_first

    Returns the fitted GLM results object (statsmodels GLMResults) and prints a brief summary.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Check that required columns exist
    required_cols = ['MajorityChoice', 'Age_centered', 'culture', 'Female', 'majority_first']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe passed to model()")

    formula = 'MajorityChoice ~ Age_centered * C(culture) + Female + majority_first'

    # Fit binomial GLM (logistic regression)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = glm_model.fit()

    # Print summary (coefficients, p-values, etc.) and return results object
    print(results.summary())

    # Optional: compute and attach marginal predicted probabilities by culture across ages
    try:
        ages = np.arange(int(df['age'].min()), int(df['age'].max()) + 1)
        pred_list = []
        cultures = sorted(df['culture'].unique())
        for c in cultures:
            for a in ages:
                row = {
                    'Age_centered': a - df['age'].mean(),
                    'culture': c,
                    'Female': df['Female'].mode()[0],
                    'majority_first': 0
                }
                pred_list.append(row)
        pred_df = pd.DataFrame(pred_list)
        pred_df['pred_prob'] = results.predict(pred_df)
        # Attach to results for downstream use
        results.predicted_by_age_culture = pred_df
    except Exception:
        # If prediction grid fails for any reason, continue without attaching
        pass

    return results


