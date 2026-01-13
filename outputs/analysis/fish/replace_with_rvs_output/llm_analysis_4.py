from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/replace_with_rvs_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing dataset into the final dataframe used for modeling.

    Produces the following additional columns used in modeling:
      - total_people: persons + child
      - person_hours: exposure = hours * total_people (used as offset)
      - fish_per_person_hour: descriptive rate = fish_caught / person_hours

    Filters out rows with missing or invalid (<= 0) hours or total_people.
    Ensures binary variables are integer 0/1.
    """
    df = df.copy()

    # Ensure numeric types where appropriate
    for col in ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with essential missing values
    df = df.dropna(subset=['fish_caught', 'hours', 'persons'])

    # Create total people (adults + children)
    df['total_people'] = df['persons'] + df['child']

    # Filter out non-positive effort or participant counts
    df = df[df['hours'] > 0]
    df = df[df['total_people'] > 0]

    # Compute person-hours exposure and descriptive rate
    df['person_hours'] = df['hours'] * df['total_people']
    # Avoid division by zero after the filters above; still guard just in case
    df['fish_per_person_hour'] = df['fish_caught'] / df['person_hours']

    # Ensure binary indicators are 0/1 integers
    df['livebait'] = df['livebait'].fillna(0).astype(int)
    df['camper'] = df['camper'].fillna(0).astype(int)

    # Keep only columns needed for modeling plus helpful descriptives
    # (model uses: fish_caught, livebait, camper, persons, child, person_hours)
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count model for fish caught using total person-hours as exposure.

    The function fits a Poisson GLM with offset = log(person_hours). It computes a dispersion
    statistic to test for overdispersion. If substantial overdispersion is detected (dispersion > 1.5),
    it fits a Negative Binomial GLM as an alternative and returns both fits.

    Returns a dictionary with:
      - 'chosen_model': 'Poisson' or 'NegativeBinomial'
      - 'poisson_model': fitted Poisson results (statsmodels object)
      - 'nb_model': fitted Negative Binomial results if fitted (otherwise None)
      - 'dispersion': Pearson chi-square / df_resid for the Poisson
      - 'formula': formula used
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    df = df.copy()

    # Formula: model the count of fish as function of predictors
    formula = 'fish_caught ~ livebait + camper + persons + child'

    # Create log offset for exposure (person-hours). Must be finite.
    df = df[df['person_hours'] > 0]
    df['log_person_hours'] = np.log(df['person_hours'])

    # Fit Poisson GLM with offset
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_person_hours']).fit()

    # Compute Pearson chi-square dispersion statistic for Poisson
    pearson_chi2 = (poisson_model.resid_pearson ** 2).sum()
    df_resid = poisson_model.df_resid
    dispersion = float(pearson_chi2 / df_resid) if df_resid > 0 else float('nan')

    results = {
        'formula': formula,
        'poisson_model': poisson_model,
        'nb_model': None,
        'dispersion': dispersion,
        'chosen_model': 'Poisson'
    }

    # If substantial overdispersion, fit Negative Binomial as alternative
    if not np.isnan(dispersion) and dispersion > 1.5:
        try:
            nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_person_hours']).fit()
            results['nb_model'] = nb_model
            results['chosen_model'] = 'NegativeBinomial'
        except Exception:
            # If NegativeBinomial fails, keep Poisson result and note it
            results['nb_model'] = None
            results['chosen_model'] = 'Poisson'

    return results

