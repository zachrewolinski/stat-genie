from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw contest dataset into a dataframe ready for modeling.

    Produces the following new columns used in modeling:
      - RelGroupSize: n_focal - n_other
      - RelLocation: dist_other - dist_focal
      - RelGroupSize_c: mean-centered RelGroupSize
      - RelLocation_c: mean-centered RelLocation
      - MaleDiff: m_focal - m_other
      - TotalSize: n_focal + n_other

    Keeps dyad and win for modeling (dyad used as a categorical fixed effect).
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing values in any required column (original NA)
    df = df.dropna(subset=required_cols)

    # Ensure numeric types for numeric columns (but leave dyad as-is for categorical use)
    numeric_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop any rows that became NA after coercion of numeric columns
    df = df.dropna(subset=numeric_cols + ['dyad'])

    # Convert dyad to a plain string/object dtype so patsy/statsmodels treat it as categorical reliably
    df['dyad'] = df['dyad'].astype(str)

    # Compute relative group size and relative location
    df['RelGroupSize'] = df['n_focal'] - df['n_other']
    df['RelLocation'] = df['dist_other'] - df['dist_focal']

    # Mean-center the predictors used in the interaction to improve interpretability
    df['RelGroupSize_c'] = df['RelGroupSize'] - df['RelGroupSize'].mean()
    df['RelLocation_c'] = df['RelLocation'] - df['RelLocation'].mean()

    # Controls
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Ensure win is numeric 0/1
    # Coercion above ensures numeric; cast to integer explicitly
    df['win'] = df['win'].astype(int)

    # Keep only columns necessary for modeling (plus helpful diagnostics columns)
    model_cols = ['win', 'RelGroupSize', 'RelLocation', 'RelGroupSize_c', 'RelLocation_c', 'MaleDiff', 'TotalSize', 'dyad',
                  'n_focal', 'n_other', 'm_focal', 'm_other', 'dist_focal', 'dist_other']
    # Some columns may not exist if original df lacked them; intersect
    model_cols = [c for c in model_cols if c in df.columns]

    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (GLM with binomial family) predicting the probability that the focal group wins
    as a function of mean-centered relative group size (RelGroupSize_c), mean-centered relative location
    (RelLocation_c), their interaction, and controls (MaleDiff, TotalSize), with dyad included as a categorical
    fixed effect to account for dyad-specific heterogeneity.

    Returns the fitted GLMResults object. Use .summary() on the result for a text summary.
    """
    # Check required model columns
    required = ['win', 'RelGroupSize_c', 'RelLocation_c', 'MaleDiff', 'TotalSize', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: main effects + interaction, controls, and dyad fixed effects
    formula = 'win ~ RelGroupSize_c * RelLocation_c + MaleDiff + TotalSize + C(dyad)'

    # Fit GLM (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return fitted model object (has attributes .params, .bse, .pvalues, .summary(), etc.)
    return model