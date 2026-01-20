from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the AMTL dataset for binomial GLM analysis.

    Produces the following columns used in modeling:
      - num_amtl: integer count of missing teeth (original)
      - sockets: integer count of observable sockets (original)
      - prop_amtl: proportion num_amtl/sockets (derived)
      - IsHuman: 1 if genus == 'Homo sapiens', else 0 (derived)
      - Age_c: age centered by the sample mean (derived)
      - ProbMale: renamed copy of prob_male (original -> standardized name)
      - tooth_class: categorical tooth class (ensured dtype)

    Rows with missing values for required fields or impossible counts are dropped.
    """
    df = df.copy()

    # Drop rows missing essential fields
    required = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
    df = df.dropna(subset=required)

    # Ensure sockets is positive integer and num_amtl is not greater than sockets
    # Convert to numeric (in case of odd types) and filter
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df = df.dropna(subset=['sockets', 'num_amtl'])
    df = df[(df['sockets'] > 0)].copy()
    df = df[df['num_amtl'] <= df['sockets']].copy()

    # Create binary indicator for Homo sapiens
    # Match exact string 'Homo sapiens' after stripping whitespace
    df['IsHuman'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Proportion of missing teeth (dependent variable for interpretation)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Center age for modeling stability
    df['Age_c'] = df['age'].astype(float) - float(df['age'].astype(float).mean())

    # Standardize prob_male column name used in model
    df['ProbMale'] = df['prob_male'].astype(float)

    # Ensure tooth_class is categorical and has consistent categories
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Final minimal set of columns used downstream (keeps extras as well)
    # Drop any remaining NA in the modeling columns
    model_cols = ['num_amtl', 'sockets', 'prop_amtl', 'IsHuman', 'Age_c', 'ProbMale', 'tooth_class']
    df = df.dropna(subset=model_cols).copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logit) generalized linear model to test whether modern humans
    (IsHuman) have higher rates of AMTL than non-human primates, adjusting for
    age, sex (probability male), and tooth class.

    The model uses the proportion prop_amtl as the response and passes sockets
    as weights so that the binomial denominator is respected (equivalent to
    modeling num_amtl ~ Binomial(sockets, p)).

    Returns the fitted GLMResults object from statsmodels.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Formula: proportion response with sockets as observation weights
    formula = 'prop_amtl ~ IsHuman + Age_c + ProbMale + C(tooth_class)'

    # Fit binomial GLM with weights equal to number of trials (sockets)
    # This treats prop_amtl as the observed proportions and sockets as frequencies
    glm_binom = smf.glm(formula=formula, data=df,
                        family=sm.families.Binomial(),
                        weights=df['sockets'])
    results = glm_binom.fit()

    # Return fitted results (caller can inspect summary(), params, conf_int(), etc.)
    return results


