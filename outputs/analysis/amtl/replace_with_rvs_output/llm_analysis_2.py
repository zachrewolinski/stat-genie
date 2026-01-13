from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for binomial regression of AMTL.

    Steps:
    - Drop rows with missing required fields
    - Drop rows with non-positive 'sockets'
    - Ensure numeric types for counts and sockets
    - Create amtl_prop = num_amtl / sockets
    - Normalize simple genus labels (e.g. 'Homo' -> 'Homo sapiens' if present) and cast to categorical
    - Cast tooth_class to categorical with a sensible ordering

    The returned dataframe contains at minimum the columns used in the model:
    ['num_amtl', 'sockets', 'amtl_prop', 'genus', 'tooth_class', 'age', 'prob_male', 'specimen']
    """
    df = df.copy()

    # Drop rows missing essential variables
    required = ['num_amtl', 'sockets', 'genus', 'tooth_class', 'age', 'prob_male', 'specimen']
    df = df.dropna(subset=required)

    # Ensure numeric types for counts and sockets
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')

    # Drop newly coerced NA rows
    df = df.dropna(subset=['num_amtl', 'sockets'])

    # Remove rows where there are no observable sockets (cannot define a binomial denominator)
    df = df[df['sockets'] > 0]

    # Proportion of missing teeth in the observable sockets (dependent variable expressed as proportion)
    df['amtl_prop'] = df['num_amtl'] / df['sockets']

    # Normalize genus strings and ensure categorical dtype
    df['genus'] = df['genus'].astype(str).str.strip()
    # Some datasets sometimes use 'Homo' shorthand; standardize to 'Homo sapiens' if needed
    df['genus'] = df['genus'].replace({'Homo': 'Homo sapiens'})
    df['genus'] = pd.Categorical(df['genus'])

    # Ensure tooth_class is a categorical with expected levels (if present in data)
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    # Provide an ordering that is reasonable; if any level is missing, pandas will handle it
    tooth_levels = ['Anterior', 'Premolar', 'Posterior']
    # If the actual data uses slightly different capitalization, we keep the original strings but attempt to coerce common variants
    df['tooth_class'] = df['tooth_class'].replace({
        'anterior': 'Anterior', 'posterior': 'Posterior', 'premolar': 'Premolar'
    })
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=tooth_levels)

    # Keep only rows that have a defined tooth_class category (if tooth_class is essential for the model)
    df = df[~df['tooth_class'].isna()]

    # Ensure age and prob_male numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df.dropna(subset=['age', 'prob_male'])

    # Final columns expected by the modeling function
    expected_cols = ['num_amtl', 'sockets', 'amtl_prop', 'genus', 'tooth_class', 'age', 'prob_male', 'specimen']
    # If other columns exist, they are preserved but we ensure expected ones are present.

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial regression testing whether modern humans ('Homo sapiens') have higher AMTL frequencies
    than other genera, adjusting for age, sex (prob_male), and tooth class. The model uses a binomial-family
    GLM on the proportion of missing teeth with the number of sockets as frequency weights. Cluster-robust
    standard errors (clustered by specimen) are requested to account for non-independence when multiple
    tooth-class observations come from the same specimen.

    Returns the fitted model results object (statsmodels GLMResults) so users can inspect coefficients, CIs, etc.
    """
    import statsmodels.formula.api as smf

    # Formula: proportion of missing teeth explained by genus (reference = Homo sapiens), tooth class, age, and prob_male
    # We explicitly set the treatment (dummy) coding reference for genus to 'Homo sapiens' so comparisons are against humans.
    formula = 'amtl_prop ~ C(genus, Treatment(reference="Homo sapiens")) + C(tooth_class) + age + prob_male'

    # Fit binomial GLM on the proportion with frequency weights equal to the number of sockets (the number of trials)
    # freq_weights will treat each observable socket as a trial so the outcome amtl_prop is the proportion of successes
    # (missing teeth) among those trials.
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])

    # Fit and request cluster-robust standard errors clustered by specimen to account for repeated measures
    # (multiple tooth classes per specimen). The returned object contains coefficient estimates; robust SEs are used
    # for inference.
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

    # Print summary for quick inspection; return the fitted results so the caller can programmatically inspect
    # coefficients, confidence intervals, predicted values, etc.
    print(results.summary())

    return results


