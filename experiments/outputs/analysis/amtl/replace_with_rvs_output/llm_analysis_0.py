from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Note: original top-level CSV read preserved if present in environment.
# If this file is imported as a module, the caller should pass their own dataframe
# to transform(). The following line is left as in the original snippet but is
# harmless if the path does not exist in the importing environment.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')
except Exception:
    # If file isn't available, we simply don't set df at module import.
    df = None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw AMTL dataset to create variables required for binomial regression.

    Output columns required by the model:
      - num_amtl: integer count of missing teeth (kept or coerced)
      - sockets: integer count of observable sockets (trials)
      - amtl_rate: proportion num_amtl / sockets
      - is_human: 1 if genus == 'Homo sapiens', else 0
      - age_z: standardized age
      - prob_male_z: standardized prob_male
      - tooth_class: categorical (kept as-is but coerced to category dtype)
      - specimen: specimen identifier (category)

    Cleaning decisions:
      - Drop rows with missing values in num_amtl, sockets, genus, age, prob_male, tooth_class, specimen.
      - Remove rows with sockets <= 0.
      - Cap num_amtl to be between 0 and sockets (if inconsistent data present).
      - After canonicalizing tooth_class, drop rows that do not map to the expected categories.
      - Clip amtl_rate slightly away from exact 0/1 for numerical stability in GLM fitting.
    """
    # Work on a copy
    df = df.copy()

    # Required columns exist
    required_cols = ['num_amtl', 'sockets', 'genus', 'age', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in core variables
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where expected
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male'])

    # Remove non-positive socket rows (cannot form a binomial trial)
    df = df[df['sockets'] > 0]

    # Ensure integer counts for num_amtl and bound them between 0 and sockets
    # Round num_amtl to nearest integer if it isn't already; then cap
    df['num_amtl'] = df['num_amtl'].round().astype(int)
    df.loc[df['num_amtl'] < 0, 'num_amtl'] = 0
    # If num_amtl > sockets, cap to sockets (data inconsistency)
    mask_over = df['num_amtl'] > df['sockets']
    if mask_over.any():
        df.loc[mask_over, 'num_amtl'] = df.loc[mask_over, 'sockets'].astype(int)

    # Create proportion column for inspection / plotting
    df['amtl_rate'] = df['num_amtl'] / df['sockets']

    # Create the main independent variable: is_human (1 for Homo sapiens, 0 otherwise)
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Standardize continuous controls (z-scores). Use ddof=0 (population sd) for stability.
    age_std = df['age'].std(ddof=0)
    prob_male_std = df['prob_male'].std(ddof=0)
    df['age_z'] = (df['age'] - df['age'].mean()) / (age_std if age_std != 0 else 1.0)
    df['prob_male_z'] = (df['prob_male'] - df['prob_male'].mean()) / (prob_male_std if prob_male_std != 0 else 1.0)

    # Ensure tooth_class is categorical and canonicalize capitalization/spacing
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().str.capitalize()

    # Map to the expected three categories. Values not in this set will become NaN and will be dropped.
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Premolar', 'Posterior'])

    # Drop rows where tooth_class did not map to one of the expected categories
    df = df[df['tooth_class'].notna()]

    # Ensure specimen is categorical for clustering
    df['specimen'] = df['specimen'].astype(str).astype('category')

    # Numerical stability: clip amtl_rate slightly away from exact 0 or 1 to avoid potential IRLS issues.
    # Keep original integer counts intact; clipping only affects the proportion used by the model.
    eps = 1e-6
    df['amtl_rate'] = df['amtl_rate'].clip(eps, 1.0 - eps)

    # Final row filter: keep only rows where amtl_rate is finite
    df = df[df['amtl_rate'].notnull() & np.isfinite(df['amtl_rate'])]

    # Return the transformed dataframe (contains all columns used in modeling)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether modern humans have higher AMTL rates than non-human primates,
    controlling for age, sex (prob_male), and tooth class. Use specimen-clustered robust standard errors
    to account for multiple observations per specimen.

    Model specification (on proportions with trial weights):
      glm( amtl_rate ~ is_human + age_z + prob_male_z + C(tooth_class),
            family=Binomial(), weights=sockets )

    Returns a dict with the fitted model result object and a tidy coefficient table including
    odds ratios and cluster-robust 95% CIs.
    """
    # Basic checks
    required_cols = ['num_amtl', 'sockets', 'amtl_rate', 'is_human', 'age_z', 'prob_male_z', 'tooth_class', 'specimen']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe must contain required columns: {missing}")

    if (df['sockets'] <= 0).any():
        raise ValueError("All 'sockets' must be > 0 for binomial modeling. Filter these before calling model().")

    # Ensure no NA in model columns
    df = df.dropna(subset=['amtl_rate', 'is_human', 'age_z', 'prob_male_z', 'tooth_class', 'specimen'])

    # Use the proportion column directly and provide sockets as weights (number of trials)
    formula = 'amtl_rate ~ is_human + age_z + prob_male_z + C(tooth_class)'

    # Convert sockets to integer weights for modeling
    weights = df['sockets'].astype(int)

    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=weights)

    # Fit model and compute specimen-clustered robust standard errors
    try:
        if 'specimen' in df.columns:
            res = glm_model.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})
        else:
            res = glm_model.fit()
    except ValueError:
        # As a fallback, attempt to fit with a very small jitter added to the dependent variable
        # to avoid boundary issues. This should be rare because transform already clips amtl_rate.
        df_jitter = df.copy()
        jitter = 1e-8
        df_jitter['amtl_rate'] = df_jitter['amtl_rate'].clip(jitter, 1.0 - jitter)
        glm_model = smf.glm(formula=formula, data=df_jitter, family=sm.families.Binomial(), weights=weights)
        if 'specimen' in df_jitter.columns:
            res = glm_model.fit(cov_type='cluster', cov_kwds={'groups': df_jitter['specimen']})
        else:
            res = glm_model.fit()

    # Build a tidy coefficient table with odds ratios and cluster-robust CIs
    params = res.params
    conf = res.conf_int()
    conf.columns = ['ci_lower', 'ci_upper']
    # Exponentiate to get odds ratios
    or_vals = np.exp(params)
    or_ci_lower = np.exp(conf['ci_lower'])
    or_ci_upper = np.exp(conf['ci_upper'])

    coef_table = pd.DataFrame({
        'term': params.index,
        'estimate_logit': params.values,
        'std_err': res.bse.values,
        'p_value': res.pvalues.values,
        'or': or_vals.values,
        'or_ci_lower': or_ci_lower.values,
        'or_ci_upper': or_ci_upper.values
    })

    # For primary inference, create a focused summary row for is_human
    if 'is_human' in params.index:
        is_human_row = coef_table[coef_table['term'] == 'is_human'].squeeze()
    else:
        is_human_row = None

    return {
        'fitted_model': res,
        'coef_table': coef_table,
        'is_human_row': is_human_row
    }