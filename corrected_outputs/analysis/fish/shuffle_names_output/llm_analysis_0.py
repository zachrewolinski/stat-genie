from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/shuffle_names_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing-visit dataframe into the analysis dataframe.

    Produces the following new columns used by the model:
      - fish_per_hour: fish_caught / hours (dependent variable)
      - log_fish_per_hour: log(1 + fish_per_hour) (auxiliary transformation)

    Ensures binary variables are coerced to 0/1 and removes invalid rows (missing fish_caught or hours, or hours <= 0).
    """
    df = df.copy()

    # Coerce expected numeric columns to numeric, introduce NaN where coercion fails
    numeric_cols = ['fish_caught', 'hours', 'persons', 'camper', 'livebait', 'child']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows without the necessary outcome / denominator
    if 'fish_caught' in df.columns and 'hours' in df.columns:
        df = df.dropna(subset=['fish_caught', 'hours'])
        # drop nonpositive hours which would make rates invalid
        df = df[df['hours'] > 0]
    else:
        # If required columns are missing, return empty dataframe with expected columns
        df = pd.DataFrame(columns=(df.columns.tolist() + ['fish_per_hour', 'log_fish_per_hour']))
        return df

    # Create fish-per-hour outcome
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Log transform (useful alternative outcome); kept as auxiliary column
    df['log_fish_per_hour'] = np.log1p(df['fish_per_hour'])

    # Ensure binary predictors are 0/1
    # Treat any value equal to 1 as 1, everything else (including >1 or NaN) as 0 for these indicator columns
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].apply(lambda x: 1 if x == 1 else 0).astype(int)
    else:
        df['livebait'] = 0

    if 'child' in df.columns:
        df['child'] = df['child'].apply(lambda x: 1 if x == 1 else 0).astype(int)
    else:
        df['child'] = 0

    # Fill missing numeric group-size/camper values with 0 (conservative) and ensure numeric dtype
    if 'persons' in df.columns:
        df['persons'] = df['persons'].fillna(0).astype(float)
    else:
        df['persons'] = 0.0

    if 'camper' in df.columns:
        df['camper'] = df['camper'].fillna(0).astype(float)
    else:
        df['camper'] = 0.0

    # Optional: remove extreme outliers in fish_per_hour if desired (here we keep all values but a user may add trimming)

    # Return only the columns required for modelling plus originals for context
    required_cols = list(df.columns)  # return full df copy; modeling code will pick needed columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear model predicting average fish caught per hour.

    Model specification:
      fish_per_hour ~ livebait * child + persons + camper

    livebait * child expands to livebait + child + livebait:child so we explicitly test whether
    the presence of a child moderates the effect of using live bait on catch rate.

    Returns:
      - statsmodels regression results object (OLS)
    """
    import statsmodels.formula.api as smf

    # Ensure the transformed variable exists
    if 'fish_per_hour' not in df.columns:
        raise ValueError("The dataframe must include a 'fish_per_hour' column. Run transform() first.")

    # Drop rows with missing predictors or outcome
    model_df = df.dropna(subset=['fish_per_hour', 'livebait', 'child', 'persons', 'camper'])

    # Fit OLS on the rate. If the outcome is highly skewed, consider using 'log_fish_per_hour' instead
    formula = 'fish_per_hour ~ livebait * child + persons + camper'
    results = smf.ols(formula=formula, data=model_df).fit()

    # Return the fitted results object (contains summary, coefficients, p-values, etc.)
    return results


