from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_and_positive_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for binomial regression of AMTL.

    Output dataframe columns used by the model:
      - num_amtl : integer count of missing teeth (successes)
      - sockets  : integer count of observable sockets (trials)
      - amtl_prop: num_amtl / sockets (proportion, used as endog in GLM)
      - genus    : categorical predictor with 'Homo sapiens' set as reference category
      - age      : original age estimate (kept for transparency)
      - age_z    : standardized age (mean 0, sd 1)
      - prob_male: continuous sex estimate (0-1)
      - tooth_class: categorical (Anterior/Premolar/Posterior)
      - specimen, pop: kept for potential diagnostics or clustering
    """
    df = df.copy()

    # required columns
    required = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class']
    df = df.dropna(subset=required)

    # Ensure sockets is positive and counts are coherent
    df = df[df['sockets'] > 0]
    # Remove rows where num_amtl is negative or greater than sockets
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Proportion (convenience column). Keep raw counts for binomial modeling.
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Standardize age for numerical stability in the model
    df['age_z'] = (df['age'] - df['age'].mean()) / (df['age'].std(ddof=0) if df['age'].std(ddof=0) != 0 else 1.0)

    # Force genus to be categorical and set reference to 'Homo sapiens'
    # If some genera are not present in the data, those categories will be NA and dropped below.
    genus_cats = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
    df['genus'] = pd.Categorical(df['genus'], categories=genus_cats)
    df = df.dropna(subset=['genus'])

    # Standardize tooth_class categories and drop any unexpected categories
    tooth_cats = ['Anterior', 'Premolar', 'Posterior']
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=tooth_cats)
    df = df.dropna(subset=['tooth_class'])

    # Keep identifier columns for diagnostics
    if 'specimen' in df.columns:
        df['specimen'] = df['specimen'].astype(str)
    if 'pop' in df.columns:
        df['pop'] = df['pop'].astype(str)

    # Final safety: ensure integer counts
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) GLM for AMTL counts with number of sockets as the number of trials.

    The model tests whether genus (with 'Homo sapiens' as reference) predicts elevated AMTL
    after adjusting for age (z-scored), prob_male, and tooth_class.

    Returns a dictionary with the fitted results object and an odds-ratio table for interpretation.
    """
    # Import inside function to ensure availability
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd

    # Formula: proportion as response; sockets passed as frequency weights (number of trials)
    # We use amtl_prop as the proportion of missing teeth and supply freq_weights=sockets
    formula = 'amtl_prop ~ C(genus) + age_z + prob_male + C(tooth_class)'

    # Fit GLM binomial using frequency weights = number of trials (sockets)
    # Note: statsmodels expects the endogenous variable to be a proportion when freq_weights are supplied
    model_glm = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])
    results = model_glm.fit()

    # Build an odds-ratio table for coefficients of interest
    params = results.params
    conf = results.conf_int()
    pvals = results.pvalues

    or_table = pd.DataFrame({
        'coef': params,
        'OR': np.exp(params),
        'CI_lower': np.exp(conf[0]),
        'CI_upper': np.exp(conf[1]),
        'pvalue': pvals
    })

    # For easier interpretation, extract genus coefficients specifically
    genus_rows = [r for r in or_table.index if r.startswith('C(genus)')]
    genus_or = or_table.loc[genus_rows].copy()

    return {
        'results': results,
        'or_table': or_table,
        'genus_or': genus_or
    }


