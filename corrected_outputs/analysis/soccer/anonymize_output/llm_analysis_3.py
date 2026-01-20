from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset to a modeling-ready dataframe with the following columns (exact names):
      - RedCards: count of red cards in the dyad (feature16)
      - Matches: number of matches in the dyad (feature9) -> used as exposure offset
      - SkinToneMean: mean of the two rater skin ratings (feature18, feature19)
      - DarkSkin: binary indicator (1 dark, 0 light) where dark if SkinToneMean >= 0.60, light if SkinToneMean <= 0.40
      - Age: player's age in years computed relative to mid-2013 season date
      - Height_cm: feature6
      - Weight_kg: feature7
      - Position: feature8 (categorical)
      - League: feature4 (categorical)
      - ImplicitBias: feature22 (country-level IAT mean)
      - ExplicitBias: feature25 (country-level explicit mean)
      - RefereeID: feature20 (used for clustering)

    The transform drops dyads with missing essential values (red cards, matches, skin ratings) and removes dyads with Matches <= 0 and keeps only players classified as dark or light (excludes middle ratings).
    """
    # Copy to avoid mutating the original
    df = df.copy()

    # Rename/bring forward the columns we need with clear names
    df = df.rename(columns={
        'feature9': 'Matches',
        'feature16': 'RedCards',
        'feature18': 'SkinRater1',
        'feature19': 'SkinRater2',
        'feature5': 'PlayerBirthdate',
        'feature6': 'Height_cm',
        'feature7': 'Weight_kg',
        'feature8': 'Position',
        'feature4': 'League',
        'feature20': 'RefereeID',
        'feature21': 'RefCountryID',
        'feature22': 'ImplicitBias',
        'feature25': 'ExplicitBias',
        'feature1': 'PlayerShortName'
    })

    # Drop rows missing essential numeric values
    df = df.dropna(subset=['RedCards', 'Matches', 'SkinRater1', 'SkinRater2'])

    # Convert counts to integers (coerce if necessary)
    df['RedCards'] = pd.to_numeric(df['RedCards'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['RedCards'])
    df['RedCards'] = df['RedCards'].astype(int)

    # Remove dyads with no exposure (zero or negative matches) because we model rates with an offset
    df['Matches'] = pd.to_numeric(df['Matches'], errors='coerce')
    df = df.dropna(subset=['Matches'])
    df = df[df['Matches'] > 0]

    # Compute mean skin tone across raters (ensure numeric)
    df['SkinRater1'] = pd.to_numeric(df['SkinRater1'], errors='coerce')
    df['SkinRater2'] = pd.to_numeric(df['SkinRater2'], errors='coerce')
    df = df.dropna(subset=['SkinRater1', 'SkinRater2'])
    df['SkinToneMean'] = (df['SkinRater1'].astype(float) + df['SkinRater2'].astype(float)) / 2.0

    # Define binary dark vs light classification. We exclude the middle/ambiguous cases.
    df['DarkSkin'] = np.where(df['SkinToneMean'] >= 0.60, 1,
                              (np.where(df['SkinToneMean'] <= 0.40, 0, np.nan)))

    # Keep only rows classified as dark (1) or light (0)
    df = df.dropna(subset=['DarkSkin'])
    df['DarkSkin'] = df['DarkSkin'].astype(int)

    # Convert birthdate to datetime and compute age at a reference date (season midpoint e.g., 2013-07-01)
    df['PlayerBirthdate'] = pd.to_datetime(df['PlayerBirthdate'], dayfirst=False, errors='coerce', infer_datetime_format=True)
    ref_date = pd.to_datetime('2013-07-01')
    df['Age'] = ((ref_date - df['PlayerBirthdate']).dt.days / 365.25).astype(float)

    # Keep height and weight as numeric
    df['Height_cm'] = pd.to_numeric(df['Height_cm'], errors='coerce')
    df['Weight_kg'] = pd.to_numeric(df['Weight_kg'], errors='coerce')

    # Drop rows with missing values in key controls after transformations
    df = df.dropna(subset=['Age', 'Height_cm', 'Weight_kg', 'Position', 'League', 'ImplicitBias', 'ExplicitBias', 'RefereeID'])

    # Ensure RefereeID is integer for clustering
    # Some RefereeID values may be non-integer strings; coerce where possible
    df['RefereeID'] = pd.to_numeric(df['RefereeID'], errors='coerce')
    df = df.dropna(subset=['RefereeID'])
    df['RefereeID'] = df['RefereeID'].astype(int)

    # Final set of columns to return (these are used in the model)
    keep_cols = ['RedCards', 'Matches', 'SkinToneMean', 'DarkSkin', 'Age', 'Height_cm', 'Weight_kg',
                 'Position', 'League', 'ImplicitBias', 'ExplicitBias', 'RefereeID']

    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression to model the rate of RedCards per Match, comparing DarkSkin to LightSkin.

    Model specification (formula):
      RedCards ~ DarkSkin + Age + Height_cm + Weight_kg + C(Position) + C(League) + ImplicitBias + ExplicitBias

    We include an offset = log(Matches) to model red cards as a rate per match. Standard errors are clustered by RefereeID to account for non-independence of dyads judged by the same referee.

    Returns a dictionary with:
      - 'model_results': clustered robust results-like object with .params and .conf_int() methods
      - 'irr': DataFrame of incidence rate ratios (exp(coef)) with 95% CIs
    """
    data = df.copy()

    # Formula uses the exact required column names
    formula = 'RedCards ~ DarkSkin + Age + Height_cm + Weight_kg + C(Position) + C(League) + ImplicitBias + ExplicitBias'

    # Offset is log of Matches
    offset = np.log(data['Matches'].astype(float))

    # Fit Negative Binomial GLM
    model_glm = sm.GLM.from_formula(formula,
                                    data=data,
                                    family=sm.families.NegativeBinomial(),
                                    offset=offset)
    fit_res = model_glm.fit()

    # Compute clustered covariance matrix by RefereeID
    # cov_cluster returns an ndarray covariance matrix
    cluster_groups = data['RefereeID'].values
    try:
        cov = cov_cluster(fit_res, cluster_groups)
    except Exception:
        # In case cov_cluster signature expects a different shape, attempt to pass as a Series
        cov = cov_cluster(fit_res, data['RefereeID'])

    # Build a simple results-like wrapper containing params, bse, cov_params() and conf_int()
    params = fit_res.params.copy()
    bse = np.sqrt(np.diag(cov))

    import types

    class ClusteredResults:
        def __init__(self, params, cov, bse):
            # params: pandas Series
            self.params = params
            self._cov = cov
            self.bse = pd.Series(bse, index=params.index)

        def cov_params(self):
            return self._cov

        def conf_int(self, alpha=0.05):
            z = 1.96  # approximate for 95% CI; using normal approximation
            lower = self.params - z * self.bse
            upper = self.params + z * self.bse
            # Return a DataFrame with columns 0 and 1 to mimic statsmodels convention
            return pd.DataFrame({0: lower, 1: upper})

    clustered = ClusteredResults(params=params, cov=cov, bse=bse)

    # Compute incidence rate ratios (IRR) and 95% CIs using clustered results
    params_ser = clustered.params
    conf = clustered.conf_int()
    irr = np.exp(params_ser)
    irr_lower = np.exp(conf[0])
    irr_upper = np.exp(conf[1])

    irr_df = pd.DataFrame({
        'coef': params_ser,
        'IRR': irr,
        'IRR_lower_95': irr_lower,
        'IRR_upper_95': irr_upper
    })

    return {
        'model_results': clustered,
        'irr': irr_df
    }