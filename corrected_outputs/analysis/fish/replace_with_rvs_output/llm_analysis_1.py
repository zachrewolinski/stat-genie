from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/replace_with_rvs_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numeric columns
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce')
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')

    # Drop rows with missing key values
    df = df.dropna(subset=['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child'])

    # Remove rows with nonpositive or extremely small hours (can't compute a rate / offset)
    df = df[df['hours'] > 0]

    # Ensure binary variables are 0/1 ints (if they are booleans or other encodings)
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Create derived variables used in modeling / diagnostics
    df['total_people'] = df['persons'] + df['child']

    # Rate variables for descriptive use
    df['fish_per_hour'] = df['fish_caught'] / df['hours']
    # Per-person-per-hour (useful for diagnostics; NaN if total_people == 0)
    df['fish_per_person_hour'] = df['fish_caught'] / (df['hours'] * df['total_people']).replace({0: np.nan})

    # Final: keep only rows with finite fish_per_hour
    df = df[np.isfinite(df['fish_per_hour'])]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Fit Poisson and Negative Binomial GLMs for count data with hours as exposure (offset = log(hours)).
    # Predictors: livebait, camper, persons, child. Dependent: fish_caught. Offset: log(hours).

    import numpy as _np
    import statsmodels.api as _sm

    # Prepare design matrix
    predictors = ['livebait', 'camper', 'persons', 'child']
    X = df[predictors].copy()
    X = _sm.add_constant(X)
    y = df['fish_caught'].astype(float)
    offset = _np.log(df['hours'].astype(float))

    # Fit Poisson (use robust SEs)
    poisson_model = _sm.GLM(y, X, family=_sm.families.Poisson(), offset=offset)
    poisson_res = poisson_model.fit(cov_type='HC0')

    # Assess overdispersion: Pearson chi2 / df_resid
    pearson_chi2 = _np.sum(poisson_res.resid_pearson ** 2)
    df_resid = poisson_res.df_resid if poisson_res.df_resid > 0 else 1
    dispersion = pearson_chi2 / df_resid

    results = {
        'poisson_result': poisson_res,
        'overdispersion_pearson_chi2': float(pearson_chi2),
        'overdispersion_ratio': float(dispersion)
    }

    # If evidence of overdispersion (dispersion >> 1), fit Negative Binomial as preferred model
    # common heuristic: dispersion > 1.5 indicates overdispersion
    if dispersion > 1.5:
        try:
            nb_model = _sm.GLM(y, X, family=_sm.families.NegativeBinomial(alpha=1.0), offset=offset)
            nb_res = nb_model.fit()
            results['negative_binomial_result'] = nb_res
            results['chosen_model'] = 'negative_binomial'
        except Exception as e:
            # If NB fails, return Poisson but note failure
            results['nb_fit_error'] = str(e)
            results['chosen_model'] = 'poisson (nb_fit_failed)'
    else:
        results['chosen_model'] = 'poisson'

    # Also compute and include estimated rate per hour (baseline) and incidence rate ratios (IRRs)
    # Compute IRRs from chosen model parameters if available
    chosen = results.get('negative_binomial_result', results['poisson_result'])
    params = chosen.params
    irr = _np.exp(params)
    conf = chosen.conf_int()
    conf_exp = _np.exp(conf)

    results['params'] = params
    results['IRR'] = irr
    results['conf_int_exp'] = conf_exp

    # Return the results dictionary (contains model objects and summary information)
    return results


