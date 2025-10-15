from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Creates:
    - masfem_z: z-scored masfem (primary IV)
    - LogDeaths: log(alldeaths + 1)
    - LogDamage: log(ndam15 + 1)
    - YearCentered: year minus mean(year)
    - Ensures key columns are numeric/categorical and drops rows missing required fields
    """
    import numpy as np
    import pandas as pd

    # Make a copy to avoid mutating input
    df = df.copy()

    # Ensure numeric fields are numeric
    numeric_cols = ['masfem', 'masfem_mturk', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure source is categorical
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Drop rows missing the critical columns needed to test the hypothesis
    required_for_analysis = ['masfem', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'year']
    # Keep only rows with at least masfem and at least one outcome and core severity metrics
    df = df.dropna(subset=required_for_analysis)

    # Create dependent variables: log transforms to reduce skew
    # Add 1 to avoid -inf for zero values
    df['LogDeaths'] = np.log(df['alldeaths'] + 1)
    df['LogDamage'] = np.log(df['ndam15'] + 1)

    # Standardize the primary IV (masfem) to aid interpretation
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) > 0 else 1.0)

    # Center year to control for temporal trends
    df['YearCentered'] = df['year'] - df['year'].mean()

    # Ensure binary gender indicator is numeric (0/1) if present
    if 'gender_mf' in df.columns:
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')

    # Keep only columns needed downstream (but preserve others if user wants them)
    # We will not drop other columns; modeling code will reference the columns listed above.

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary statistical models to evaluate the relationship between name femininity and outcomes:
    1) Negative binomial GLM for raw death counts (alldeaths) to model count nature and overdispersion.
    2) OLS regression for LogDamage (log(ndam15 + 1)) to model economic impact.

    Both models include the same covariates and an interaction between masfem_z and gender_mf to test moderation.

    Returns a dictionary with fitted model results objects.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Ensure required columns exist
    required = ['alldeaths', 'LogDeaths', 'LogDamage', 'masfem_z', 'wind', 'min', 'category', 'YearCentered', 'source']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula string; include interaction with gender_mf if available
    base_predictors = 'masfem_z + wind + min + category + YearCentered + masfem_mturk'
    if 'gender_mf' in df.columns:
        formula_terms = base_predictors + ' + gender_mf + masfem_z:gender_mf + C(source)'
    else:
        formula_terms = base_predictors + ' + C(source)'

    # 1) Negative binomial GLM for counts of deaths
    # Use alldeaths (raw counts) as outcome and include an offset if exposure were available; here we model counts directly.
    formula_nb = 'alldeaths ~ ' + formula_terms
    try:
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
        results['nb_deaths'] = nb_model
    except Exception as e:
        # If NB fails, fall back to Poisson with robust covariance
        pois_model = smf.glm(formula=formula_nb, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
        results['poisson_deaths_robust'] = pois_model

    # 2) OLS for log damages
    formula_ols = 'LogDamage ~ ' + formula_terms
    ols_model = smf.ols(formula=formula_ols, data=df).fit(cov_type='HC3')
    results['ols_damage'] = ols_model

    # Return the fitted result objects so the caller can inspect .summary(), params, conf_int(), etc.
    return results


