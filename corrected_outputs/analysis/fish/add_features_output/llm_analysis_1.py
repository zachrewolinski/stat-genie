from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing dataset to the analytic dataframe used for modelling.

    Produces/returns at least the following columns used by the model:
      - fish_caught (dependent count)
      - hours (exposure)
      - livebait (0/1)
      - camper (0/1)
      - group_size_c (centered group size = persons + child)
      - prop_children (child / group_size)
      - religiousness_c (centered)
      - year_c (centered)
      - lunch_c (centered)
      - county (categorical)
      - fish_per_hour (derived, for descriptive use)

    Rows with missing or invalid hours or fish_caught are dropped.
    """

    df = df.copy()

    # Drop rows missing core outcome or exposure
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove rows with non-positive hours (cannot be used as exposure)
    df = df[df['hours'] > 0]

    # Ensure binary indicators are integers 0/1
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].astype('Int64').fillna(0).astype(int)
    else:
        df['livebait'] = 0
    if 'camper' in df.columns:
        df['camper'] = df['camper'].astype('Int64').fillna(0).astype(int)
    else:
        df['camper'] = 0

    # Construct group size and proportion children
    # Use available columns: persons (adults) and child (children)
    df['persons'] = df['persons'].fillna(0)
    df['child'] = df['child'].fillna(0)
    df['group_size'] = df['persons'] + df['child']

    # Protect against division by zero: if group_size is 0 (shouldn't be), set prop_children to 0
    df['prop_children'] = df.apply(lambda r: (r['child'] / r['group_size']) if r['group_size'] > 0 else 0.0, axis=1)

    # Derived descriptive variable: fish per hour
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Center continuous covariates to aid interpretation
    df['group_size_c'] = df['group_size'] - df['group_size'].mean()

    # If year/lunch/religiousness present, center them; otherwise create NA-filled centered columns
    if 'year' in df.columns:
        df['year_c'] = df['year'].astype(float) - df['year'].astype(float).mean()
    else:
        df['year_c'] = 0.0

    if 'lunch' in df.columns:
        df['lunch_c'] = df['lunch'].astype(float) - df['lunch'].astype(float).mean()
    else:
        df['lunch_c'] = 0.0

    if 'religiousness' in df.columns:
        df['religiousness_c'] = df['religiousness'].astype(float) - df['religiousness'].astype(float).mean()
    else:
        df['religiousness_c'] = 0.0

    # Make county explicitly categorical if available (helps modeling with C(county))
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')
    else:
        # create a placeholder single-level category so formula code doesn't error
        df['county'] = 'unknown'
        df['county'] = df['county'].astype('category')

    # Final required columns check: ensure types are numeric where needed
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')

    # Drop any rows introduced with NaNs in the numeric conversions
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Keep the transformed columns and original useful columns
    # (Model will reference the columns listed in the conceptual variables)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count regression models to estimate factors associated with fish caught per hour.

    Approach:
      - Fit a Poisson GLM with log(hours) as an offset to model fish_caught as a rate (fish per hour).
      - Compute dispersion = deviance / df_resid. If dispersion > ~1.5 this indicates overdispersion.
      - Fit a Negative Binomial GLM as a more robust model to overdispersion.
      - Return both fitted models and diagnostics; recommend the Negative Binomial if overdispersion is present.

    Returns a dict containing:
      - 'poisson': fitted Poisson results (statsmodels object)
      - 'neg_bin': fitted Negative Binomial results (statsmodels object)
      - 'dispersion': dispersion statistic from the Poisson
      - 'use_negative_binomial': boolean flag (True if dispersion > 1.5)

    Notes: this function prints model summaries for quick inspection and also returns the fitted objects
    so the caller can inspect coefficients, confidence intervals, AIC, etc.
    """

    # Work on a copy to avoid side-effects
    df = df.copy()

    # Required columns check
    required = ['fish_caught', 'hours', 'livebait', 'camper', 'group_size_c', 'prop_children',
                'religiousness_c', 'year_c', 'lunch_c', 'county']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare offset (log of exposure hours)
    offset = np.log(df['hours'].astype(float))

    # Base formula: include key independent variables and controls; county as a factor
    formula = ('fish_caught ~ livebait + camper + group_size_c + prop_children + '
               'religiousness_c + year_c + lunch_c + C(county)')

    # Fit Poisson GLM with robust (HC0) covariance to be conservative
    poisson_model = sm.GLM.from_formula(formula,
                                       data=df,
                                       family=sm.families.Poisson(),
                                       offset=offset)
    poisson_res = poisson_model.fit(cov_type='HC0')

    # Dispersion diagnostic (Poisson deviance / df_resid)
    dispersion = poisson_res.deviance / poisson_res.df_resid if poisson_res.df_resid != 0 else np.nan

    # Fit Negative Binomial GLM (uses statsmodels NegativeBinomial family). Also use robust SEs.
    nb_model = sm.GLM.from_formula(formula,
                                   data=df,
                                   family=sm.families.NegativeBinomial(),
                                   offset=offset)
    nb_res = nb_model.fit(cov_type='HC0')

    # Print concise summaries to console for quick inspection
    try:
        print("Poisson model summary (robust SE):")
        print(poisson_res.summary())
    except Exception:
        pass
    try:
        print("Negative Binomial model summary (robust SE):")
        print(nb_res.summary())
    except Exception:
        pass

    results = {
        'poisson': poisson_res,
        'neg_bin': nb_res,
        'dispersion': dispersion,
        'use_negative_binomial': bool(dispersion > 1.5)
    }

    return results


