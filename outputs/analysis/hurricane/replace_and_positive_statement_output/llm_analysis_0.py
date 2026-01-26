from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_and_positive_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into variables used for modeling.

    Produces/keeps the following columns required by the model:
      - alldeaths: integer count (dependent variable)
      - masfem_std: standardized femininity score (IV)
      - gender_female: binary female-name indicator (IV)
      - severity_z: standardized severity index constructed from wind and min pressure (control)
      - elapsedyrs: passed through from original data (control)
      - year: passed through from original data (control)
      - source: categorical source variable (control)
      - log_ndam15: log(1 + ndam15) (auxiliary, not used in primary model but kept for robustness checks)

    Notes:
      - Rows with missing values for the core variables are dropped.
      - Standardization uses sample mean and sample std (ddof=0 for numeric stability here).
    """
    df = df.copy()

    # Core columns required for the analysis - drop rows missing these
    required = ['alldeaths', 'masfem', 'wind', 'min', 'gender_mf', 'elapsedyrs', 'year']
    df = df.dropna(subset=required)

    # Dependent variable: ensure integer count
    # (alldeaths should already be integer/count; cast defensively)
    df['alldeaths'] = df['alldeaths'].astype(int)

    # Independent variables
    # Binary female-name indicator (0/1)
    df['gender_female'] = df['gender_mf'].astype(int)

    # Standardize continuous femininity rating
    df['masfem_std'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Construct a severity index from wind (higher = stronger) and min pressure (lower = stronger)
    # Convert min pressure to 'neg_min' so higher neg_min means stronger storm
    df['neg_min'] = -df['min']
    # z-scores
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1.0)
    df['neg_min_z'] = (df['neg_min'] - df['neg_min'].mean()) / (df['neg_min'].std(ddof=0) if df['neg_min'].std(ddof=0) != 0 else 1.0)
    # Average the two z-scores to create a single severity index
    df['severity_z'] = (df['wind_z'] + df['neg_min_z']) / 2.0

    # Controls: ensure categorical source is a string and fill missing
    df['source'] = df['source'].fillna('unknown').astype(str)

    # Auxiliary transformation: logged damage (useful for robustness checks)
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        df['log_ndam15'] = np.nan

    # Keep only columns needed (plus a few useful extras for diagnostics)
    keep_cols = [
        'ind', 'year', 'name', 'masfem', 'masfem_std', 'gender_mf', 'gender_female',
        'min', 'neg_min', 'wind', 'severity_z', 'category', 'alldeaths', 'ndam', 'ndam15', 'log_ndam15',
        'elapsedyrs', 'source'
    ]
    # Some columns may not exist in all dataset variants; intersect with existing columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a generalized linear model appropriate for count data (negative binomial) to estimate
    the association between hurricane name femininity and fatalities, controlling for storm
    severity and temporal/source covariates.

    Model formula (primary):
      alldeaths ~ masfem_std + gender_female + severity_z + elapsedyrs + year + C(source)

    We choose a Negative Binomial GLM to allow for overdispersion common in count data.
    Robust (HC3) standard errors are used to reduce sensitivity to heteroskedasticity.

    Returns the fitted model object (statsmodels results) after printing a summary.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['alldeaths', 'masfem_std', 'gender_female', 'severity_z', 'elapsedyrs', 'year', 'source']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: include C(source) to treat 'source' as categorical
    formula = 'alldeaths ~ masfem_std + gender_female + severity_z + elapsedyrs + year + C(source)'

    # Fit Negative Binomial GLM
    # Note: statsmodels' GLM supports families.NegativeBinomial
    model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
    results = model.fit(cov_type='HC3')

    # Print summary for quick inspection
    print(results.summary())

    # Recommended robustness checks (not executed here, left as guidance):
    #  - Fit Poisson and compare dispersion; test sensitivity to adding log population/exposure if available.
    #  - Replace alldeaths with log1p(alldeaths) and run OLS as a semi-parametric robustness check.
    #  - Use ndam15 (damage) or log_ndam15 as alternative outcomes.
    #  - Test models using masfem (unstandardized) and masfem_mturk as alternative femininity measures.

    return results


