from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/replace_with_rvs_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used by the statistical model.

    Produces/keeps the following columns required by the model:
      - nuts_opened: (int) count outcome (keeps original column name)
      - seconds: (float) session duration in seconds (keeps original column name)
      - log_seconds: (float) natural log of seconds (used as offset/exposure)
      - age: (float) age in years (keeps original column name)
      - sex_m: (int) sex indicator (1 = male, 0 = female)
      - help_y: (int) help indicator (1 = received help, 0 = no help)
      - hammer: (category) hammer type (kept as-is for use as a factor)
      - chimpanzee: (int or category) individual ID (kept as-is for use as a factor)
      - nuts_per_sec: (float) auxiliary: nuts_opened / seconds (for diagnostics/plotting)

    Drops rows with missing or invalid values for essential fields.
    """
    # Make a copy to avoid mutating caller's dataframe
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Coerce numeric columns
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Normalize categorical columns to strings for reliable mapping
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    df['help'] = df['help'].astype(str).str.strip().str.lower()
    df['hammer'] = df['hammer'].astype(str).astype('category')
    # Keep chimpanzee as-is but ensure no missing
    df['chimpanzee'] = df['chimpanzee']

    # Drop rows with missing essential values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Remove impossible or degenerate session durations
    df = df[df['seconds'] > 0]

    # Binary encodings used in modeling
    df['sex_m'] = (df['sex'] == 'm').astype(int)
    # Accept common yes tokens (y or yes), treat others as no
    df['help_y'] = df['help'].isin(['y', 'yes']).astype(int)

    # Exposure / offset: log(seconds)
    # Make sure seconds strictly positive (we already filtered seconds > 0)
    df['log_seconds'] = np.log(df['seconds'].astype(float))

    # Derived diagnostic variable: efficiency as nuts per second
    df['nuts_per_sec'] = df['nuts_opened'] / df['seconds']

    # Final sanity filter: nuts_opened non-negative integer (or zero)
    df = df[df['nuts_opened'] >= 0]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a count model for nuts opened with session duration as exposure.

    Modeling strategy:
      1) Fit a Poisson GLM with offset = log_seconds.
      2) Compute a simple overdispersion metric: Pearson chi2 / df_resid.
      3) If overdispersion > 1.5, refit with a Negative Binomial GLM.

    The model formula includes main effects for age, sex, and help plus interactions of help with age and sex,
    and controls for hammer type and chimpanzee ID as categorical fixed effects.

    Returns the fitted results object (statsmodels result instance) for downstream inspection.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure the offset column exists
    if 'log_seconds' not in df.columns:
        raise ValueError('Transformed dataframe must contain column "log_seconds" (log of seconds)')

    # Formula: count outcome predicted by age, sex indicator, help indicator, interactions, and controls
    formula = 'nuts_opened ~ age + sex_m + help_y + help_y:age + help_y:sex_m + C(hammer) + C(chimpanzee)'

    # Fit Poisson first
    poisson_mod = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_seconds'])
    poisson_res = poisson_mod.fit()

    # Overdispersion measure: Pearson chi2 / df_resid
    try:
        pearson_chi2 = poisson_res.pearson_chi2
        df_resid = poisson_res.df_resid
        overdispersion = pearson_chi2 / float(df_resid) if df_resid > 0 else np.nan
    except Exception:
        # If attribute missing for some reason, set to NaN
        overdispersion = np.nan

    # Decide whether to use Negative Binomial
    if not np.isfinite(overdispersion) or overdispersion <= 1.5:
        results = poisson_res
        results.model_choice = 'Poisson'
        results.overdispersion = overdispersion
    else:
        # Fit Negative Binomial as remedy for overdispersion
        negbin_mod = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_seconds'])
        negbin_res = negbin_mod.fit()
        negbin_res.model_choice = 'NegativeBinomial'
        negbin_res.overdispersion = overdispersion
        results = negbin_res

    # Return the fitted model object (caller can print .summary())
    return results


