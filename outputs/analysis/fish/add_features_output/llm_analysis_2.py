from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Defensive copy
    df = df.copy()

    # Required columns for analysis
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']

    # Drop rows missing the required core variables
    df = df.dropna(subset=required_cols)

    # Ensure numeric types for binary indicators
    for col in ['livebait', 'camper', 'persons', 'child', 'fish_caught', 'hours']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # After coercion, drop any newly-NA required rows
    df = df.dropna(subset=required_cols)

    # Remove or fix rows with nonpositive hours (can't take log of nonpositive exposure)
    # If hours are extremely small but >0 that's OK. Remove zeros or negative values.
    df = df[df['hours'] > 0]

    # Create derived variables
    # group_size: total people in the visiting group (adults 'persons' + children 'child')
    df['group_size'] = df['persons'] + df['child']

    # children_binary: indicator for presence of >=1 child
    df['children_binary'] = (df['child'] > 0).astype(int)

    # Create rate variable as a descriptive column (fish per hour)
    df['rate_fish_per_hour'] = df['fish_caught'] / df['hours']

    # Exposure log for use as an offset in count models
    # Use natural log; safe because we have filtered hours > 0
    df['exposure_log'] = np.log(df['hours'])

    # Ensure categorical columns are of appropriate type
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Keep only columns necessary for subsequent modelling and reporting
    keep_cols = [
        'fish_caught', 'hours', 'exposure_log', 'rate_fish_per_hour',
        'livebait', 'camper', 'persons', 'child', 'group_size', 'children_binary',
        'religiousness', 'age', 'county'
    ]
    # Some of these may be missing in input; keep those that exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model for fish_caught with hours as exposure (offset = log(hours)).
    Strategy:
      1. Fit Poisson GLM with offset.
      2. Compute Pearson dispersion statistic. If dispersion > 1.5 (substantial overdispersion) fit Negative Binomial.
      3. Return the chosen fitted model (statsmodels result object). Also print summaries for diagnostics.

    Model formula (main specification):
      fish_caught ~ livebait + camper + group_size + children_binary + religiousness + age + C(county)
    Only terms available in df are included. 'C(county)' will add county fixed effects if county exists.
    """
    # Build formula using columns present in the dataframe
    predictors = []
    if 'livebait' in df.columns:
        predictors.append('livebait')
    if 'camper' in df.columns:
        predictors.append('camper')
    if 'group_size' in df.columns:
        predictors.append('group_size')
    if 'children_binary' in df.columns:
        predictors.append('children_binary')
    if 'religiousness' in df.columns:
        predictors.append('religiousness')
    if 'age' in df.columns:
        predictors.append('age')
    # include county fixed effects if available
    if 'county' in df.columns:
        predictors.append('C(county)')

    if len(predictors) == 0:
        raise ValueError('No predictors available in the dataframe to fit the model.')

    formula = 'fish_caught ~ ' + ' + '.join(predictors)

    # Ensure exposure_log exists
    if 'exposure_log' not in df.columns:
        raise ValueError("exposure_log column required for offset (log(hours)). Run the transform function first.)")

    # Fit Poisson GLM
    poisson_model = sm.GLM.from_formula(formula, data=df, family=sm.families.Poisson(),
                                        offset=df['exposure_log']).fit()

    # Compute Pearson chi2 dispersion (measure of overdispersion)
    mu = poisson_model.fittedvalues
    y = poisson_model.model.endog
    # Avoid division by zero in Pearson calculation (mu should be >0 for Poisson predictions)
    eps = 1e-8
    pearson_chi2 = np.sum(((y - mu) ** 2) / np.maximum(mu, eps))
    df_resid = poisson_model.df_resid if hasattr(poisson_model, 'df_resid') else max(len(y) - poisson_model.df_model - 1, 1)
    dispersion = pearson_chi2 / df_resid

    print('Poisson model fitted. Pearson chi2 = {:.2f}, df_resid = {:.0f}, dispersion = {:.2f}'.format(
        pearson_chi2, df_resid, dispersion))

    # Choose Negative Binomial if overdispersion is substantial
    chosen_model = poisson_model
    if dispersion > 1.5:
        print('Overdispersion detected (dispersion > 1.5). Fitting Negative Binomial GLM.')
        try:
            nb_model = sm.GLM.from_formula(formula, data=df, family=sm.families.NegativeBinomial(),
                                           offset=df['exposure_log']).fit()
            chosen_model = nb_model
            print('Negative Binomial model fitted.')
        except Exception as e:
            print('Negative Binomial model failed to converge; returning Poisson model. Error:', e)
            chosen_model = poisson_model
    else:
        print('No strong overdispersion detected; retaining Poisson model.')

    # Print chosen model summary
    try:
        print(chosen_model.summary())
    except Exception:
        # summary may raise if model object is not standard; ignore
        pass

    # Return the chosen fitted model object so caller can inspect results programmatically
    results = chosen_model
    return results


