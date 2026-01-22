from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/negative_leading_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataframe into analysis-ready form. Creates efficiency measures and binary predictors.

    Output columns used in the model:
      - efficiency: nuts_opened / seconds (continuous DV)
      - log_efficiency: log1p(efficiency) (alternative DV if needed)
      - age_c: centered age (IV)
      - sex_m: 1 for male, 0 for female (IV)
      - help_y: 1 if received help, 0 otherwise (IV)
      - hammer: kept as categorical control
      - chimpanzee: kept for grouping (random effect)
    """
    df = df.copy()

    # Drop rows missing critical outcome or time information
    df = df.dropna(subset=['nuts_opened', 'seconds'])

    # Remove or flag non-positive session durations to avoid division by zero
    df = df[df['seconds'] > 0]

    # Create efficiency (nuts opened per second)
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Create a log-transformed efficiency as a robust alternative (useful if skewed)
    # Use log1p to handle zero efficiencies safely
    df['log_efficiency'] = np.log1p(df['efficiency'])

    # Standardize / center age for interpretability
    # If age has missing values, keep them as NaN (they will be dropped by model-fitting if necessary)
    if 'age' in df.columns:
        df['age_c'] = df['age'] - df['age'].mean()
    else:
        df['age_c'] = np.nan

    # Convert sex to binary indicator: male = 1, female = 0
    # Accept common letter cases and handle unexpected categories as NaN
    def sex_map(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        if x in ['m', 'male']:
            return 1
        if x in ['f', 'female']:
            return 0
        return np.nan

    df['sex_m'] = df['sex'].apply(sex_map)

    # Convert help to binary indicator: yes = 1, no = 0
    def help_map(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        if x in ['y', 'yes']:
            return 1
        if x in ['n', 'no']:
            return 0
        return np.nan

    df['help_y'] = df['help'].apply(help_map)

    # Ensure hammer and chimpanzee are treated as categorical / grouping
    df['hammer'] = df['hammer'].astype('category')
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # Optionally, drop rows missing the main IVs (age, sex, help) or keep and let model handle missingness
    # We'll drop rows missing the key predictors to keep analyses straightforward (small sample size considerations)
    df = df.dropna(subset=['age_c', 'sex_m', 'help_y', 'efficiency'])

    # Reset index for clean output
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects model (primary) and an OLS model (secondary) to test whether age, sex, and receiving help
    influence nut-cracking efficiency.

    Primary model: Mixed effects model with random intercepts for chimpanzee to account for repeated measures.
    Formula: efficiency ~ age_c + sex_m + help_y + C(hammer)

    Secondary model: OLS with the same fixed effects (for comparison and to provide standard regression outputs).

    Returns a dictionary with fitted model results objects and printed summaries.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Ensure the dataframe has the columns we expect
    required = ['efficiency', 'age_c', 'sex_m', 'help_y', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Fit mixed effects model (random intercepts for chimpanzee)
    formula = 'efficiency ~ age_c + sex_m + help_y + C(hammer)'
    try:
        md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)  # use ML for easier comparisons
        print('\nMixedLM fit summary:')
        print(mdf.summary())
        results['mixedlm'] = mdf
    except Exception as e:
        # If the mixed model fails (e.g., convergence issues due to small n), capture the error and proceed to OLS
        print('MixedLM failed with error:', e)
        results['mixedlm_error'] = str(e)

    # Fit plain OLS for fixed-effect estimates as a robustness check
    ols_formula = formula
    ols_model = smf.ols(ols_formula, data=df).fit()
    print('\nOLS fit summary:')
    print(ols_model.summary())
    results['ols'] = ols_model

    # Additional diagnostics / simple group summaries to help evaluate the 'No' hypothesis
    # Means and standard errors by sex and help
    group_stats = df.groupby(['sex_m', 'help_y'])['efficiency'].agg(['mean', 'std', 'count']).reset_index()
    print('\nGroup summaries (efficiency) by sex_m and help_y:')
    print(group_stats)
    results['group_stats'] = group_stats

    # Return results dict (fitted models and diagnostics)
    return results


