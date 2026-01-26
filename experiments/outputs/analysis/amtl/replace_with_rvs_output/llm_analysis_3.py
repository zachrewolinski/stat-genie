from typing import Any
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw AMTL dataset into a modeling-ready dataframe.

    Returns dataframe with the following columns required for modeling:
      - num_amtl: integer count of missing teeth (successes)
      - sockets: integer count of observable sockets (trials)
      - genus: categorical genus (keeps original strings)
      - tooth_class: categorical tooth class
      - specimen: specimen identifier (kept as category)
      - age_c: mean-centered age
      - prob_male: probability specimen is male (0-1)
      - amtl_rate: num_amtl / sockets (diagnostic)
    """

    # Work on a copy
    df = df.copy()

    # Required columns check (will raise KeyError if missing)
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in input dataframe")

    # Ensure numeric types for counts
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce').astype('Float64')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce').astype('Float64')

    # Remove rows with missing essential data
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Remove impossible or degenerate rows: non-positive sockets or num_amtl > sockets or negative values
    df = df[df['sockets'] > 0]
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Convert counts to integer type (safe after filtering)
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)

    # Create diagnostic rate column
    df['amtl_rate'] = df['num_amtl'] / df['sockets']

    # Center age for modeling stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure prob_male within [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df[(df['prob_male'] >= 0) & (df['prob_male'] <= 1)]

    # Ensure categorical columns are of dtype 'category' and strip whitespace
    df['genus'] = df['genus'].astype(str).str.strip()
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    df['specimen'] = df['specimen'].astype(str).str.strip()

    df['genus'] = df['genus'].astype('category')
    df['tooth_class'] = df['tooth_class'].astype('category')
    df['specimen'] = df['specimen'].astype('category')

    # Filter to genera of interest (defensive; dataset description lists Homo sapiens, Pan, Pongo, Papio)
    allowed_genera = set(['Homo sapiens', 'Pan', 'Pongo', 'Papio'])
    df = df[df['genus'].isin(allowed_genera)]

    # Final reset index
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM modeling AMTL frequency (num_amtl out of sockets) as a function of genus
    while controlling for age (centered), sex probability (prob_male), and tooth class.

    Returns the fitted model results with specimen-clustered robust standard errors.

    Modeling approach:
      - Use the proportion response (num_amtl / sockets) with weights = sockets in a Binomial GLM.
      - Set the reference level for genus to 'Homo sapiens' so coefficients for other genera are comparisons to modern humans.
      - Cluster robust standard errors on 'specimen' to account for non-independence within specimens.
    """

    # Work with a copy to avoid side-effects
    df = df.copy()

    # Ensure the dataframe contains required columns
    for col in ['num_amtl', 'sockets', 'genus', 'age_c', 'prob_male', 'tooth_class', 'specimen']:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not present in dataframe passed to model()")

    # Defensive coercions to correct dtypes for modeling
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce').astype(int)
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce').astype(int)
    df['age_c'] = pd.to_numeric(df['age_c'], errors='coerce')
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
    df = df.dropna(subset=['num_amtl', 'sockets', 'age_c', 'prob_male', 'genus', 'tooth_class', 'specimen'])

    # Create an adjusted proportion to avoid exact 0 or 1 which can cause boundary problems in Binomial GLM.
    # This is an internal helper column and is NOT a replacement of the conceptual variables num_amtl/sockets.
    # Use a simple Agresti-Coull type adjustment: (x + 0.5) / (n + 1)
    df['amtl_prop_adj'] = (df['num_amtl'].astype(float) + 0.5) / (df['sockets'].astype(float) + 1.0)

    # Build genus term, preferring to set Homo sapiens as the reference if present
    genus_values = df['genus'].astype(str).unique()
    if 'Homo sapiens' in genus_values:
        genus_term = 'C(genus, Treatment(reference="Homo sapiens"))'
    else:
        # If Homo sapiens is not present in the data, fall back to default coding.
        genus_term = 'C(genus)'

    formula = f'amtl_prop_adj ~ {genus_term} + age_c + prob_male + C(tooth_class)'

    # Fit GLM binomial with frequency weights equal to number of trials (sockets).
    # Use freq_weights (the canonical keyword in statsmodels formula API) and an adjusted response to avoid boundary NaNs.
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['sockets'])

    try:
        result = glm_model.fit()
    except ValueError:
        # As a fallback, try a more robust fit with a small number of iterations and the 'newton' method,
        # which sometimes avoids the initial-deviance NaN issue. If this still fails, re-raise.
        try:
            result = glm_model.fit(method='newton', maxiter=100, disp=0)
        except Exception:
            # If fitting still fails, raise the original error to surface the problem.
            raise

    # Obtain cluster-robust standard errors clustered on specimen
    try:
        clustered_res = result.get_robustcov_results(cov_type='cluster', groups=df['specimen'])
    except Exception:
        # If clustering fails for some reason, fall back to the original result
        clustered_res = result

    # Return both the raw fit and the clustered-SE result for full inspection
    return {
        'glm_result': result,
        'glm_result_clustered_se': clustered_res
    }