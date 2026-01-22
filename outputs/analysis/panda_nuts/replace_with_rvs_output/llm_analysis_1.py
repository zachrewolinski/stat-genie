from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_with_rvs_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw chimpanzee nut-cracking dataset into a dataframe suitable for mixed-effects modeling.

    Produces the following columns required by the model:
      - LogRate: log((nuts_opened + 0.5) / seconds)
      - age: copied from original
      - sex_M: binary indicator 1 if male, 0 if female
      - Help: binary indicator 1 if help == 'y'/'yes', 0 if 'N'/'n' or similar
      - ChimpID: string identifier for grouping (from 'chimpanzee')
      - hammer_*: dummy variables (drop-first) for hammer type (prefixed with 'hammer_')

    Rows with missing values for essential columns are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Required raw columns
    required_cols = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing core data
    df = df.dropna(subset=['chimpanzee', 'age', 'sex', 'nuts_opened', 'seconds', 'help'])

    # Ensure numeric types
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')

    # Drop if conversion produced NaNs in essential numeric columns
    df = df.dropna(subset=['age', 'nuts_opened', 'seconds'])

    # Create grouping id for random effects
    df['ChimpID'] = df['chimpanzee'].astype(str)

    # Create binary sex indicator: sex_M = 1 if male, 0 if female
    # handle capitalization and possible variations
    df['sex_M'] = df['sex'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['m', 'male'] else 0)

    # Create Help indicator: 1 if help in ('y','yes'), otherwise 0
    df['Help'] = df['help'].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ['y', 'yes'] else 0)

    # Compute rate of nuts opened per second; add small pseudocount to avoid log(0)
    # Use +0.5 continuity correction on counts, a common small-sample adjustment
    df['Rate'] = (df['nuts_opened'].astype(float) + 0.5) / df['seconds'].astype(float)

    # Log-transform the rate for modeling (continuous outcome)
    # Guard against non-positive values by clipping a very small positive floor
    eps = 1e-8
    df['Rate'] = df['Rate'].clip(lower=eps)
    df['LogRate'] = np.log(df['Rate'])

    # Dummy-encode hammer type; drop_first=True to avoid multicollinearity
    hammer_dummies = pd.get_dummies(df['hammer'].astype(str).str.strip(), prefix='hammer', drop_first=True)
    # If there are no dummies (e.g., single hammer type), this will be an empty dataframe
    if not hammer_dummies.empty:
        df = pd.concat([df.reset_index(drop=True), hammer_dummies.reset_index(drop=True)], axis=1)

    # Final set of columns we want to keep for modeling; keep extras if present (hammer dummies)
    # Ensure the dataframe contains at least the core model columns
    model_cols = ['LogRate', 'age', 'sex_M', 'Help', 'ChimpID']
    # Add hammer dummy columns if present
    hammer_cols = [c for c in df.columns if c.startswith('hammer_')]
    model_cols += hammer_cols

    # Return only the columns needed for modeling (plus any hammer dummies)
    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log-rate of nuts opened per second (LogRate)
    from age, sex, and help, controlling for hammer type and with a random intercept for ChimpID.

    Returns the fitted mixed-effects model result object.
    """
    import statsmodels.formula.api as smf

    # Validate presence of required columns
    required = ['LogRate', 'age', 'sex_M', 'Help', 'ChimpID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Identify hammer dummy columns (if any)
    hammer_cols = [c for c in df.columns if c.startswith('hammer_')]

    # Build formula
    base_terms = ['age', 'sex_M', 'Help']
    all_terms = base_terms + hammer_cols
    formula = 'LogRate ~ ' + ' + '.join(all_terms)

    # Fit mixed effects model with random intercept by ChimpID
    # Use REML=False for likelihood-based comparisons (common default for inference)
    md = smf.mixedlm(formula, data=df, groups=df['ChimpID'])
    mdf = md.fit(reml=False)

    # Return the fitted model object (contains .summary(), params, etc.)
    return mdf


