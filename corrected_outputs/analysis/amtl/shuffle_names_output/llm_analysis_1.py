from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with the exact columns required for modeling.

    The function produces the required final columns:
      - 'is_human' (0/1 indicator for Homo sapiens)
      - 'num_missing' (integer count of missing teeth in the scored tooth class)
      - 'n_sockets' (integer count of observable sockets scored)
      - 'age_at_death' (numeric)
      - 'sex_male_prob' (numeric 0-1 or other coded value)
      - 'tooth_class' (categorical)

    Additional helper columns (e.g., 'genus_label') may be present but are not
    required by the model directly.
    """
    df = df.copy()

    # Safely access potential input columns; use None/defaults when absent
    stdev_age_col = df.get('stdev_age', pd.Series(index=df.index, dtype='float64'))
    prob_male_col = df.get('prob_male', pd.Series(index=df.index, dtype='float64'))
    num_amtl_col = df.get('num_amtl', pd.Series(index=df.index, dtype='float64'))
    pop_col = df.get('pop', pd.Series(index=df.index, dtype='float64'))

    # num_missing: take from 'stdev_age' (rounded to integer)
    df['num_missing'] = pd.to_numeric(stdev_age_col, errors='coerce').round()

    # n_sockets: take from 'prob_male' (rounded to integer)
    df['n_sockets'] = pd.to_numeric(prob_male_col, errors='coerce').round()

    # age_at_death: take from 'num_amtl' (numeric, can be fractional)
    df['age_at_death'] = pd.to_numeric(num_amtl_col, errors='coerce')

    # sex_male_prob: take from 'pop'
    df['sex_male_prob'] = pd.to_numeric(pop_col, errors='coerce')

    # tooth_class: prefer 'genus' per schema mapping; fall back to existing 'tooth_class' column if needed
    if 'genus' in df.columns:
        df['tooth_class'] = df['genus'].astype(str)
    else:
        # ensure we have a column to work with
        tooth_col = df.get('tooth_class', pd.Series(index=df.index, dtype='object'))
        df['tooth_class'] = tooth_col.astype(str)

    # Derive genus_label from 'age' per provided schema mapping for creating is_human
    age_col = df.get('age', pd.Series(index=df.index, dtype='object'))
    df['genus_label'] = age_col.astype(str).str.strip()

    # is_human indicator: 1 if genus_label equals 'Homo sapiens' (case-insensitive)
    df['is_human'] = (df['genus_label'].str.lower() == 'homo sapiens').astype(int)

    # Ensure numeric types for counts before rounding/casting
    df['num_missing'] = pd.to_numeric(df['num_missing'], errors='coerce')
    df['n_sockets'] = pd.to_numeric(df['n_sockets'], errors='coerce')

    # Drop rows missing essential count information
    df = df[df['n_sockets'].notnull() & df['num_missing'].notnull()]

    # Round and cast counts to integers
    df['n_sockets'] = df['n_sockets'].round().astype(int)
    df['num_missing'] = df['num_missing'].round().astype(int)

    # Enforce logical bounds on counts
    df.loc[df['num_missing'] < 0, 'num_missing'] = 0
    # Cap num_missing at n_sockets
    mask_over = df['num_missing'] > df['n_sockets']
    df.loc[mask_over, 'num_missing'] = df.loc[mask_over, 'n_sockets']

    # Remove rows where no sockets are observed (cannot model binomial)
    df = df[df['n_sockets'] > 0]

    # Convert tooth_class to categorical, with missing as 'unknown'
    df['tooth_class'] = df['tooth_class'].replace({None: 'unknown', 'nan': 'unknown'})
    df['tooth_class'] = df['tooth_class'].fillna('unknown').astype('category')

    # Ensure the final required columns exist (create with NA if absent)
    required_final = ['is_human', 'num_missing', 'n_sockets', 'age_at_death', 'sex_male_prob', 'tooth_class']
    for c in required_final:
        if c not in df.columns:
            df[c] = pd.NA

    # Optionally keep helper columns for traceability
    final_cols = ['num_missing', 'n_sockets', 'is_human', 'age_at_death', 'sex_male_prob', 'tooth_class', 'genus_label']
    # If 'specimen' exists in source, include it; otherwise add NA so callers expecting it don't error
    if 'specimen' in df.columns:
        final_cols.insert(0, 'specimen')
    else:
        df['specimen'] = pd.NA
        final_cols.insert(0, 'specimen')

    return df[final_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression for AMTL using a GLM with binomial family.

    Model specification:
      - Response: proportion missing (num_missing / n_sockets) with frequency weights = n_sockets
      - Predictors: is_human (main predictor), age_at_death (continuous control), sex_male_prob (continuous control), and tooth_class (categorical control)

    Returns the fitted GLM results object (statsmodels). Prints the model summary.
    """
    # Work on a copy to avoid modifying caller data
    df = df.copy()

    # Ensure essential columns exist
    essential = ['num_missing', 'n_sockets', 'is_human', 'tooth_class']
    missing_ess = [c for c in essential if c not in df.columns]
    if missing_ess:
        raise ValueError(f"Missing essential columns for modeling: {missing_ess}")

    # Drop rows missing essential modeling values
    df_model = df.dropna(subset=essential).copy()

    # Ensure counts are integers and n_sockets positive
    df_model['n_sockets'] = pd.to_numeric(df_model['n_sockets'], errors='coerce').astype(int)
    df_model['num_missing'] = pd.to_numeric(df_model['num_missing'], errors='coerce').astype(int)
    df_model = df_model[df_model['n_sockets'] > 0]

    # Recompute proportion and clip to avoid exact 0/1 which can cause numerical issues in GLM with logit link
    df_model['prop'] = df_model['num_missing'] / df_model['n_sockets']
    eps = 1e-4
    df_model['prop'] = df_model['prop'].clip(eps, 1 - eps)

    # Ensure tooth_class is treated as categorical
    df_model['tooth_class'] = df_model['tooth_class'].astype('category')

    # If after filtering there is no data, raise a clear error
    if df_model.shape[0] == 0:
        raise ValueError("No observations available for modeling after filtering and preprocessing.")

    # Construct formula: use 'prop' as response and frequency weights (n_sockets) for number of trials
    formula = 'prop ~ is_human + age_at_death + sex_male_prob + C(tooth_class)'

    # Fit GLM with binomial family using frequency weights = n_sockets
    try:
        glm_binom = smf.glm(formula=formula, data=df_model,
                            family=sm.families.Binomial(),
                            weights=df_model['n_sockets'])
        results = glm_binom.fit()
    except Exception:
        # If fitting still fails, attempt a more stable fit by providing start_params via a simple intercept-only initialization
        try:
            glm_binom = smf.glm(formula=formula, data=df_model,
                                family=sm.families.Binomial(),
                                weights=df_model['n_sockets'])
            start_params = np.zeros(glm_binom.exog.shape[1])
            results = glm_binom.fit(start_params=start_params, maxiter=100, disp=0)
        except Exception as e:
            # Re-raise with context if still failing
            raise RuntimeError("GLM binomial failed to converge or returned numerical errors.") from e

    # Print summary for quick inspection
    print(results.summary())

    return results