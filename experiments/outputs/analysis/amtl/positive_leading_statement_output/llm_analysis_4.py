from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/positive_leading_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid modifying original
    df = df.copy()

    # Essential columns required for binomial modeling
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
    # Drop rows with missing essential information
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove impossible rows (no observable sockets or negative counts)
    df = df[df['sockets'] > 0]
    df = df[df['num_amtl'] >= 0]

    # Cap num_amtl to sockets (in case of recording/rounding issues)
    df['num_amtl'] = df[['num_amtl', 'sockets']].apply(lambda row: min(int(round(row['num_amtl'])), int(round(row['sockets']))), axis=1)

    # Create the complementary count (failures) for binomial modeling
    df['num_non_amtl'] = df['sockets'] - df['num_amtl']

    # Create a binary indicator for modern humans (Homo sapiens)
    # Normalize textual variations if present
    df['genus'] = df['genus'].astype(str)
    df['is_human'] = (df['genus'].str.strip().str.lower() == 'homo sapiens').astype(int)

    # Ensure tooth_class is categorical with consistent capitalization
    df['tooth_class'] = df['tooth_class'].astype(str).str.title()
    df.loc[~df['tooth_class'].isin(['Anterior', 'Posterior', 'Premolar']), 'tooth_class'] = pd.NA
    df = df.dropna(subset=['tooth_class'])

    # Center continuous controls to improve model stability/interpretability
    df['age_c'] = df['age'] - df['age'].mean()
    df['prob_male_c'] = df['prob_male'] - df['prob_male'].mean()

    # Keep only columns necessary for modeling and downstream checks
    keep_cols = ['specimen', 'genus', 'is_human', 'pop', 'tooth_class',
                 'num_amtl', 'num_non_amtl', 'sockets', 'age', 'age_c', 'prob_male', 'prob_male_c']
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    # Final sanity checks: remove any rows where counts are negative after processing
    df = df[(df['num_amtl'] >= 0) & (df['num_non_amtl'] >= 0)]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits a binomial GLM (logistic link) to test whether modern humans (is_human==1)
    have higher AMTL frequency than non-human primates, controlling for age,
    sex (prob_male), and tooth class.

    Returns a dictionary with the fitted model results object, a table of
    exponentiated coefficients (odds ratios) and their 95% CIs, and the model summary text.
    """
    # Work on a copy
    df = df.copy()

    # Build design matrix: use is_human, centered age and prob_male, and tooth_class dummies
    # tooth_class dummies: drop_first=True to use the first alphabetically or naturally as reference.
    # We'll explicitly set 'Anterior' as reference by using drop_first after ordering categories.
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Posterior', 'Premolar'], ordered=False)

    # Create dummy variables for tooth_class, dropping the reference 'Anterior'
    tooth_dummies = pd.get_dummies(df['tooth_class'], prefix='tooth', drop_first=True)

    # Explanatory variables
    exog = pd.concat([
        df[['is_human', 'age_c', 'prob_male_c']].astype(float).reset_index(drop=True),
        tooth_dummies.reset_index(drop=True)
    ], axis=1)

    # Add intercept
    exog = sm.add_constant(exog, has_constant='add')

    # Endogenous: 2-column array (num_successes, num_failures) for Binomial family
    endog = np.column_stack((df['num_amtl'].astype(int), df['num_non_amtl'].astype(int)))

    # Fit GLM (binomial) with logit link
    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    results = model.fit()

    # Extract coefficients, compute odds ratios and 95% CIs
    params = results.params
    conf = results.conf_int()
    or_series = np.exp(params)
    or_ci = np.exp(conf)

    or_table = pd.DataFrame({
        'coef': params,
        'odds_ratio': or_series,
        'ci_lower': or_ci[0],
        'ci_upper': or_ci[1]
    })

    # Specifically report the effect of is_human
    human_row = or_table.loc['is_human'] if 'is_human' in or_table.index else None

    # Prepare a compact summary
    summary_text = results.summary().as_text()

    return {
        'results': results,
        'odds_ratio_table': or_table,
        'is_human_summary': human_row,
        'summary_text': summary_text
    }


