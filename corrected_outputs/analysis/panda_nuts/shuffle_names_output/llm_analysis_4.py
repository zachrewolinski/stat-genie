from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/shuffle_names_output/panda_nuts.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataframe into a cleaned dataframe with the exact columns used in the model.

    Assumptions based on the provided schema (several columns are mislabelled):
      - 'seconds' contains an individual ID.
      - 'nuts_opened' actually contains age in years.
      - 'age' contains sex ('f'/'m').
      - 'help' contains the number of nuts opened in the session.
      - 'sex' contains duration of the session in seconds.
      - 'chimpanzee' contains whether help was received ('y'/'N').

    This function creates the following columns used in modeling:
      - ID (int)
      - age_years (float)
      - sex (category-like string 'M'/'F')
      - hammer (string)
      - nuts_opened (int)
      - duration_seconds (float)
      - received_help (binary 1/0)
      - efficiency_nps (nuts per second)
      - log_efficiency (log of efficiency)
    """
    df = df.copy()

    # Capture raw source columns before we overwrite any target column names
    raw_id_col = df.get('seconds', pd.Series(np.nan, index=df.index))
    raw_age_col = df.get('nuts_opened', pd.Series(np.nan, index=df.index))  # maps to age_years
    raw_sex_col = df.get('age', pd.Series(np.nan, index=df.index))  # maps to sex
    raw_hammer_col = df.get('hammer', pd.Series(np.nan, index=df.index))
    raw_nuts_opened_col = df.get('help', pd.Series(np.nan, index=df.index))  # maps to nuts_opened
    raw_duration_col = df.get('sex', pd.Series(np.nan, index=df.index))  # maps to duration_seconds
    raw_chimpanzee_col = df.get('chimpanzee', pd.Series(np.nan, index=df.index))  # maps to received_help

    # ID: prefer 'seconds' column if present, otherwise use index
    id_num = pd.to_numeric(raw_id_col, errors='coerce')
    # fill any non-convertible IDs with the dataframe index to ensure no missing IDs
    id_num = id_num.fillna(pd.Series(df.index, index=df.index))
    # make integer IDs
    # guard against floats that cannot be represented as ints (use astype after rounding)
    df['ID'] = id_num.round().astype(int)

    # Age in years (originally labelled 'nuts_opened' in file schema)
    df['age_years'] = pd.to_numeric(raw_age_col, errors='coerce')

    # Sex (originally labelled 'age' in file schema).
    # Normalize to 'M'/'F' where possible. Treat unknown/missing as NaN.
    sex_str = raw_sex_col.astype(str).str.strip().str.lower()
    sex_str = sex_str.replace({'nan': np.nan, 'none': np.nan, '': np.nan})
    def map_sex(x):
        if not isinstance(x, str):
            return np.nan
        x = x.strip().lower()
        if x.startswith('f'):
            return 'F'
        if x.startswith('m'):
            return 'M'
        return np.nan
    df['sex'] = sex_str.map(map_sex)

    # Hammer type: coerce to string and provide 'unknown' for missing values
    hammer_str = raw_hammer_col.astype(str).str.strip()
    hammer_str = hammer_str.replace({'nan': np.nan, 'none': np.nan, '': np.nan})
    df['hammer'] = hammer_str.fillna('unknown')

    # Number of nuts opened in session (originally labelled 'help' in file schema)
    df['nuts_opened'] = pd.to_numeric(raw_nuts_opened_col, errors='coerce')

    # Duration of session in seconds (originally labelled 'sex' in file schema)
    df['duration_seconds'] = pd.to_numeric(raw_duration_col, errors='coerce')

    # Received help from another chimpanzee: map 'y'/'Y'/'yes' -> 1, 'n'/'N'/'no' -> 0
    help_str = raw_chimpanzee_col.astype(str).str.strip().str.lower()
    help_str = help_str.replace({'nan': np.nan, 'none': np.nan, '': np.nan})
    def map_help(x):
        # handle numeric 1/0 already present
        if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
            if x == 1:
                return 1
            if x == 0:
                return 0
            # otherwise fall through to try string handling
        if not isinstance(x, str):
            return np.nan
        x = x.strip().lower()
        if x.startswith('y'):
            return 1
        if x.startswith('n'):
            return 0
        return np.nan
    df['received_help'] = help_str.map(map_help)

    # --- Clean: drop rows missing essential variables ---
    # Include 'sex' as essential because it's an IV in the conceptual variables.
    essential = ['age_years', 'nuts_opened', 'duration_seconds', 'received_help', 'sex']
    df = df.dropna(subset=essential)

    # Filter out impossible/zero values that would break efficiency calculation
    df = df[(df['nuts_opened'] > 0) & (df['duration_seconds'] > 0)]

    # Compute efficiency: nuts per second (higher is better)
    df['efficiency_nps'] = df['nuts_opened'] / df['duration_seconds']

    # Log-transform of efficiency for robustness (kept for diagnostics / alternative models)
    df['log_efficiency'] = np.log(df['efficiency_nps'] + 1e-9)

    # Ensure correct dtypes for modeling: keep sex and hammer as strings (object)
    df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
    # received_help should be numeric
    df['received_help'] = pd.to_numeric(df['received_help'], errors='coerce').astype(float)
    # ID already int
    df['ID'] = df['ID'].astype(int)
    # hammer and sex remain as object/string types
    df['hammer'] = df['hammer'].astype(str)
    # sex should be 'M'/'F' strings; cast to object and ensure no 'nan' string remains
    df['sex'] = df['sex'].astype('object')

    # Keep only the columns needed for downstream modeling
    out_cols = ['ID', 'age_years', 'sex', 'hammer', 'nuts_opened', 'duration_seconds', 'received_help', 'efficiency_nps', 'log_efficiency']
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects model to estimate how age, sex, and receiving help influence nut-cracking efficiency.

    Primary model: mixed-effects model with random intercept for ID to account for repeated measures per individual.
    Formula: efficiency_nps ~ age_years + C(sex) + received_help + C(hammer)

    If the mixed model fails to converge, fall back to an OLS with cluster-robust SEs clustered by ID.

    Returns the fitted results object.
    """
    # Ensure necessary columns exist
    required = ['efficiency_nps', 'age_years', 'sex', 'received_help', 'hammer', 'ID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop any remaining NA in modeling columns
    mdf = df.dropna(subset=required).copy()

    # If no data remain after filtering, raise a clear error
    if mdf.shape[0] == 0:
        raise ValueError("No observations available for modeling after dropping missing values. "
                         "Check the transform() output and input data.")

    # Check that categorical variables have at least one non-missing level
    for cat in ['sex', 'hammer']:
        if mdf[cat].dropna().nunique() == 0:
            raise ValueError(f"Categorical variable '{cat}' has no observed levels in the data; cannot fit model.")

    formula = 'efficiency_nps ~ age_years + C(sex) + received_help + C(hammer)'

    # Try mixed-effects model with random intercept for ID
    try:
        md = sm.MixedLM.from_formula(formula, groups=mdf['ID'], data=mdf)
        mres = md.fit(reml=False)
        return mres
    except Exception:
        # Fall back to OLS with cluster-robust standard errors by ID
        ols = smf.ols(formula, data=mdf).fit()
        # attach clustered standard errors as an attribute for inspection
        try:
            clustered_se = ols.get_robustcov_results(cov_type='cluster', groups=mdf['ID'])
            return clustered_se
        except Exception:
            # if clustering fails, return the plain OLS result
            return ols