from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for rate (fish per hour) modeling.

    Produces the following columns (used in modeling):
      - fish_caught (int/float): count outcome (keeps original values)
      - hours (float): hours spent in park (original column, validated)
      - log_hours (float): natural log of hours (offset)
      - livebait (int 0/1): binary predictor
      - camper (int 0/1): binary predictor
      - persons (int): number of adults
      - child (int): number of children
      - group_size (int): persons + child
      - county (category): categorical control

    Rows with missing or invalid essential values (fish_caught, hours, livebait,
    camper, persons, child, county) are dropped. Hours <= 0 are dropped (cannot
    take log). Extremely small positive hours are clipped for numerical stability
    when computing log_hours; the original hours column is preserved.
    """
    df = df.copy()

    # Ensure essential columns exist
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child', 'county']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Coerce numeric columns
    for col in ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing essential values (including county)
    df = df.dropna(subset=['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child', 'county'])

    # Remove nonpositive hours (cannot take log); these rows cannot be used to estimate rate
    df = df[df['hours'] > 0]

    # Make binary indicators integer 0/1 (clip any unexpected values)
    # Coerce to numeric first to avoid weird category types
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce').fillna(0).astype(int).clip(0, 1)
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce').fillna(0).astype(int).clip(0, 1)

    # Ensure persons and child are integer counts (round if necessary)
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce').round().fillna(0).astype(int)
    df['child'] = pd.to_numeric(df['child'], errors='coerce').round().fillna(0).astype(int)

    # Derived columns
    df['group_size'] = df['persons'] + df['child']

    # Compute log_hours for offset.
    # For numerical stability in GLM fitting, clip hours used to compute the log to a small positive epsilon.
    # Keep the original 'hours' column (unmodified) but compute log_hours from a clipped version.
    eps = 1e-3  # 0.001 hours ~ 3.6 seconds; conservative lower bound for numerical stability
    hours_for_log = df['hours'].astype(float).replace([np.inf, -np.inf], np.nan)
    # Any NaNs here indicate problematic rows; drop them
    hours_for_log = hours_for_log.dropna()
    # Align df to hours_for_log (shouldn't drop anything because of earlier dropna and hours>0)
    df = df.loc[hours_for_log.index].copy()
    # Clip to epsilon to avoid extremely large negative logs that can cause underflow in exp()
    hours_clipped = hours_for_log.clip(lower=eps)
    df['log_hours'] = np.log(hours_clipped)

    # Ensure no non-finite values remain in log_hours
    finite_mask = np.isfinite(df['log_hours'])
    if not finite_mask.all():
        df = df.loc[finite_mask].copy()

    # County as categorical (keeps original labels). Modeling code will use C(county) to include fixed effects.
    df['county'] = df['county'].astype('category')

    # Drop negative counts in fish_caught (shouldn't exist). Keep large counts.
    df = df[df['fish_caught'] >= 0]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression for fish_caught with an offset for log_hours to estimate fish-per-hour rates.

    Steps:
      1) Fit a Poisson GLM with offset = log_hours and predictors: livebait, camper, persons, child, group_size, and county fixed effects.
      2) Compute a simple overdispersion statistic (Pearson chi-square / df). If that ratio > 1.5, fit a Negative Binomial GLM (NB2) as a robustness check.

    Returns a dict with keys:
      - 'poisson_result': fitted Poisson result object
      - 'overdispersion': pearson_chi2 / df_resid (float)
      - 'nb_result': fitted NB result object if fitted else None
    """
    import patsy
    from statsmodels.discrete.discrete_model import NegativeBinomial as DiscreteNegativeBinomial

    # Validate required columns present
    required_cols = ['fish_caught', 'log_hours', 'livebait', 'camper', 'persons', 'child', 'group_size', 'county']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for model: {missing}")

    # Ensure no missing or non-finite values in critical columns used by the model
    critical = df[required_cols].copy()
    # Coerce numeric where expected
    for col in ['fish_caught', 'log_hours', 'livebait', 'camper', 'persons', 'child', 'group_size']:
        critical[col] = pd.to_numeric(critical[col], errors='coerce')
    # Drop rows with any non-finite values in critical columns
    is_finite = np.ones(len(critical), dtype=bool)
    for col in ['fish_caught', 'log_hours', 'livebait', 'camper', 'persons', 'child', 'group_size']:
        is_finite &= np.isfinite(critical[col].values)
    # Also ensure fish_caught is non-negative
    is_finite &= (critical['fish_caught'].values >= 0)
    if not is_finite.all():
        df = df.loc[is_finite].reset_index(drop=True)

    # Formula: include county fixed effects via C(county)
    formula = 'fish_caught ~ livebait + camper + persons + child + group_size + C(county)'

    # Prepare offset as a finite numpy array
    offset_array = np.asarray(df['log_hours'].astype(float))
    if not np.isfinite(offset_array).all():
        raise ValueError("Non-finite values found in log_hours offset after cleaning.")

    # To improve numerical stability in IRLS (avoid initial mu values of exactly 0 or extremely large),
    # center the offset by subtracting its mean. This shifts the intercept but leaves other coefficients unchanged.
    offset_centered = offset_array - np.mean(offset_array)

    # Fit Poisson with offset (centered)
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=offset_centered)
    # Increase maxiter to give IRLS a better chance to converge in edge cases
    poisson_res = poisson_model.fit(maxiter=100, disp=False)

    # Compute Pearson chi-square / df_resid as a simple overdispersion diagnostic
    resid_pearson = poisson_res.resid_pearson
    pearson_chi2 = np.sum(resid_pearson**2)
    df_resid = poisson_res.df_resid if hasattr(poisson_res, 'df_resid') else max(len(df) - getattr(poisson_res, 'df_model', 0) - 1, 1)
    overdispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

    nb_res = None
    # If substantial overdispersion present, fit Negative Binomial (NB2)
    if not np.isnan(overdispersion) and overdispersion > 1.5:
        try:
            # Try family-based NegativeBinomial first (use same centered offset)
            nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset_centered)
            nb_res = nb_model.fit(maxiter=100, disp=False)
        except Exception:
            # If the family-based NB fails, fall back to discrete NegativeBinomial from statsmodels.discrete
            try:
                # Build design matrices using patsy via formula API (this will include the intercept)
                y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')
                # Convert y to 1d array
                y = np.asarray(y).ravel()
                # Ensure X has a constant
                if 'Intercept' not in X.columns and 'const' not in X.columns:
                    X = sm.add_constant(X, prepend=True)
                # Fit discrete NB2 (loglike_method='nb2')
                nb_disc = DiscreteNegativeBinomial(y, X, loglike_method='nb2')
                nb_res = nb_disc.fit(disp=False, maxiter=100)
            except Exception:
                nb_res = None

    # Return results and diagnostic
    return {
        'poisson_result': poisson_res,
        'overdispersion': overdispersion,
        'nb_result': nb_res
    }