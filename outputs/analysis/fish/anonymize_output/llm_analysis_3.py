from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataset into a dataframe ready for modeling.

    Produces the following columns (used in the model):
      - FishCaught: integer count of fish caught (from feature1)
      - LiveBait: binary indicator (0/1) from feature2
      - Camper: binary indicator (0/1) from feature3
      - NumAdults: integer from feature4
      - NumChildren: integer from feature5
      - Hours: positive float from feature6 (exposure)
      - TotalPeople: derived NumAdults + NumChildren
      - FishPerHour: descriptive rate FishCaught / Hours

    Rows with missing or invalid Hours (<= 0) are removed because Hours is used as exposure.
    """
    # work on a copy
    df = df.copy()

    # Rename columns to expressive names used downstream, but only if those raw columns exist.
    rename_map = {
        'feature1': 'FishCaught',
        'feature2': 'LiveBait',
        'feature3': 'Camper',
        'feature4': 'NumAdults',
        'feature5': 'NumChildren',
        'feature6': 'Hours'
    }
    # Only include mappings where the source column exists to avoid accidental creation of NaNs
    effective_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if effective_rename:
        df = df.rename(columns=effective_rename)

    # Required final column names (must exist in the final dataframe)
    req_cols = ['FishCaught', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'Hours']

    # Ensure all required columns exist in the dataframe (create with NaN if missing)
    for col in req_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Cast to numeric types where appropriate (coerce errors -> NaN)
    df['FishCaught'] = pd.to_numeric(df['FishCaught'], errors='coerce')
    df['LiveBait'] = pd.to_numeric(df['LiveBait'], errors='coerce')
    df['Camper'] = pd.to_numeric(df['Camper'], errors='coerce')
    df['NumAdults'] = pd.to_numeric(df['NumAdults'], errors='coerce')
    df['NumChildren'] = pd.to_numeric(df['NumChildren'], errors='coerce')
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce')

    # Drop rows missing the absolutely required modeling columns: FishCaught and Hours
    # (Hours is exposure and FishCaught is the dependent variable).
    df = df.dropna(subset=['FishCaught', 'Hours'])

    # Drop rows with non-positive Hours because Hours is used as exposure (log(Hours) required)
    df = df[df['Hours'] > 0]

    # If no rows remain, return an empty dataframe with the required columns and derived columns present
    if df.shape[0] == 0:
        # create empty dataframe with required and derived columns
        out_cols = req_cols + ['TotalPeople', 'FishPerHour']
        empty_df = pd.DataFrame(columns=out_cols)
        # ensure proper dtypes for consistency
        empty_df = empty_df.astype({
            'FishCaught': 'Int64',
            'LiveBait': 'Int64',
            'Camper': 'Int64',
            'NumAdults': 'Int64',
            'NumChildren': 'Int64',
            'Hours': 'float64',
            'TotalPeople': 'Int64',
            'FishPerHour': 'float64'
        })
        return empty_df

    # For other required predictors, fill missing values with sensible defaults (0)
    # These are control or indicator variables where 0 is a conservative default.
    df['LiveBait'] = df['LiveBait'].fillna(0)
    df['Camper'] = df['Camper'].fillna(0)
    df['NumAdults'] = df['NumAdults'].fillna(0)
    df['NumChildren'] = df['NumChildren'].fillna(0)

    # Cast integer-like columns to integers (round first to handle floats)
    df['FishCaught'] = df['FishCaught'].round().astype(int)

    # For binary indicators, ensure 0/1: round then clip between 0 and 1
    df['LiveBait'] = df['LiveBait'].round().astype(int).clip(0, 1)
    df['Camper'] = df['Camper'].round().astype(int).clip(0, 1)

    df['NumAdults'] = df['NumAdults'].round().astype(int)
    df['NumChildren'] = df['NumChildren'].round().astype(int)

    # Derived variables
    df['TotalPeople'] = df['NumAdults'] + df['NumChildren']

    # Descriptive rate (not used directly as DV in model but useful for checks)
    df['FishPerHour'] = df['FishCaught'] / df['Hours']

    # Reset index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression model for FishCaught using park-hours as exposure.

    Model specification:
      - Family: Negative Binomial (to accommodate overdispersion relative to Poisson)
      - Offset: log(Hours) to model rate per hour (i.e., log expected count = log(Hours) + linear predictors)
      - Predictors: LiveBait, Camper, NumAdults, NumChildren

    Returns a dictionary with the fitted model object and an incidence-rate-ratio (IRR) table.
    If the input dataframe has zero rows, returns {'results': None, 'irr_table': empty DataFrame}.
    """
    # work on a copy to avoid side-effects
    df = df.copy()

    # Verify required columns exist
    required = ['FishCaught', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'Hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns for modeling: {missing}")

    # If dataframe is empty, return an empty result structure (do not raise)
    if df.shape[0] == 0:
        irr_table = pd.DataFrame(columns=['coef', 'IRR', 'IRR_ci_lower', 'IRR_ci_upper', 'pvalue'])
        return {
            'results': None,
            'irr_table': irr_table
        }

    # Ensure Hours are positive (required for log offset)
    if (df['Hours'] <= 0).any():
        raise ValueError("All 'Hours' values must be positive to use as an exposure offset.")

    # Ensure there are no missing values in required columns
    if df[required].isnull().any().any():
        raise ValueError("Input dataframe contains missing values in required columns. Please run transform() to prepare the data.")

    # Basic formula - predictors chosen as factors likely to affect catch and effort
    formula = 'FishCaught ~ LiveBait + Camper + NumAdults + NumChildren'

    # Prepare offset = log(Hours) as a Series to preserve alignment
    offset = np.log(df['Hours'])

    # Fit GLM with Negative Binomial family
    try:
        model_glm = sm.GLM.from_formula(formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
        results = model_glm.fit()
    except Exception:
        # fallback to Poisson and then attempt to obtain robust standard errors
        model_glm = sm.GLM.from_formula(formula, data=df, family=sm.families.Poisson(), offset=offset)
        results = model_glm.fit()
        try:
            # convert to robust covariance results
            results = results.get_robustcov_results(cov_type='HC0')
        except Exception:
            # if that fails, keep the plain fit results
            pass

    # Prepare incidence rate ratios (IRRs) and 95% CIs
    params = results.params
    conf = results.conf_int()
    irr = np.exp(params)
    # conf may be a DataFrame with two columns; use iloc to be robust
    irr_ci_lower = np.exp(conf.iloc[:, 0])
    irr_ci_upper = np.exp(conf.iloc[:, 1])

    irr_table = pd.DataFrame({
        'coef': params,
        'IRR': irr,
        'IRR_ci_lower': irr_ci_lower,
        'IRR_ci_upper': irr_ci_upper,
        'pvalue': results.pvalues
    })

    # Order columns for readability
    irr_table = irr_table[['coef', 'IRR', 'IRR_ci_lower', 'IRR_ci_upper', 'pvalue']]

    return {
        'results': results,
        'irr_table': irr_table
    }