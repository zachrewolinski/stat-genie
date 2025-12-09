from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid side effects
    df = df.copy()

    # Keep only rows with the primary variables present
    required_cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source', 'gender_mf']
    df = df.dropna(subset=required_cols)

    # Dependent variable: fatalities as integer count
    # Ensure non-negative integer counts; if fractional or negative, coerce sensibly
    df['alldeaths_count'] = df['alldeaths'].astype(float).clip(lower=0)
    # Round to nearest integer if necessary (fatalities should be integer counts)
    df['alldeaths_count'] = df['alldeaths_count'].round().astype(int)

    # Independent variable: masfem (higher = more feminine name)
    # Create a standardized (z-scored) version for interpretability in regression
    df['masfem'] = df['masfem'].astype(float)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Ensure gender_mf is binary numeric 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Center year to improve interpretability and numerical stability
    df['year_centered'] = df['year'].astype(float) - df['year'].astype(float).mean()

    # Ensure category is an integer (1-5) used as categorical in modeling
    df['category'] = df['category'].astype(int)

    # elapsedyrs keep as numeric
    df['elapsedyrs'] = df['elapsedyrs'].astype(float)

    # Ensure wind and min are numeric
    df['wind'] = df['wind'].astype(float)
    df['min'] = df['min'].astype(float)

    # Create an auxiliary logged DV for alternative linear modeling (not used in main NB model but useful)
    df['log_alldeaths_plus1'] = np.log1p(df['alldeaths_count'])

    # Keep only columns necessary for modeling and diagnostics
    keep_cols = [
        'alldeaths_count', 'log_alldeaths_plus1',
        'masfem', 'masfem_z', 'gender_mf',
        'wind', 'min', 'category', 'year_centered', 'elapsedyrs', 'source', 'name', 'year'
    ]

    # Some sources may have many categories; keep as-is (we will model as categorical)
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Work on a copy
    data = df.copy()

    # Primary model: Negative Binomial GLM for count outcome (fatalities)
    # Formula includes masfem standardized (masfem_z) as the main predictor and
    # controls for physical intensity (wind, min, category), temporal trend (year_centered),
    # archival effects (source), and elapsedyrs. gender_mf is included as an additional control.
    formula = (
        'alldeaths_count ~ masfem_z + gender_mf + wind + min + C(category) '
        '+ year_centered + elapsedyrs + C(source)'
    )

    # Fit the GLM Negative Binomial
    glm_nb = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial())
    res_nb = glm_nb.fit()

    # Obtain robust (HC3) covariance estimates to be conservative about heteroskedasticity
    try:
        res_nb_robust = res_nb.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fallback: return the original results if robust covariance calculation fails
        res_nb_robust = res_nb

    # Secondary / robustness checks (returned in a dict):
    # 1) OLS on log(alldeaths + 1)
    ols_formula = (
        'log_alldeaths_plus1 ~ masfem_z + gender_mf + wind + min + C(category) '
        '+ year_centered + elapsedyrs + C(source)'
    )
    ols_res = smf.ols(formula=ols_formula, data=data).fit()

    # 2) A Poisson GLM (check for overdispersion vs NB)
    pois_res = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()

    # Return a dictionary of fitted results so the caller can inspect each model
    results = {
        'nb_glm_robust': res_nb_robust,
        'nb_glm': res_nb,
        'poisson_glm': pois_res,
        'ols_log_outcome': ols_res,
        'model_formula': formula
    }

    return results


