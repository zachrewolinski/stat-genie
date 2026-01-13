from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/replace_with_rvs_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for modeling the effect of reader_view on speed with dyslexia as moderator.

    Transformations performed:
    - Copy input dataframe to avoid side-effects
    - Drop rows missing key variables (speed, reader_view, dyslexia_bin, uuid)
    - Create explicit device dummy columns (device_smartphone, device_tablet) so column names are stable
    - Map english_native to a binary column english_native_bin (1 = Y, 0 otherwise)
    - Ensure dyslexia_bin is integer (0/1)
    - Standardize numerical covariates num_words, Flesch_Kincaid, age -> num_words_z, Flesch_Kincaid_z, age_z
    - Create interaction column reader_view_x_dyslexia
    - Return dataframe containing all columns needed for modeling
    """
    df = df.copy()

    # Required columns
    required_cols = ['speed', 'reader_view', 'dyslexia_bin', 'uuid']
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in input dataframe")

    # Drop rows missing primary variables
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin', 'uuid'])

    # Ensure numeric types for key columns
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)
    # dyslexia_bin is present in schema; if not numeric, coerce
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)

    # Create interaction term
    df['reader_view_x_dyslexia'] = df['reader_view'] * df['dyslexia_bin']

    # Device dummies (explicit mapping so column names are stable)
    if 'device' in df.columns:
        df['device_smartphone'] = (df['device'] == 'smartphone').astype(int)
        df['device_tablet'] = (df['device'] == 'tablet').astype(int)
    else:
        # If device column missing, create zero columns
        df['device_smartphone'] = 0
        df['device_tablet'] = 0
    # desktop is the omitted category (both dummies = 0)

    # Map english_native to binary
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].astype(str).str.strip().str.upper().eq('Y').astype(int)
    else:
        # if column missing, create zeros (non-native) to avoid errors; user may choose to drop instead
        df['english_native_bin'] = 0

    # Ensure retake_trial numeric
    if 'retake_trial' in df.columns:
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Standardize continuous covariates to aid model convergence
    def zscore_col(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors='coerce')
        m = s.mean()
        sd = s.std()
        if pd.isna(sd) or sd == 0:
            return (s - m).fillna(0)
        return ((s - m) / sd).fillna(0)

    # num_words
    if 'num_words' in df.columns:
        df['num_words_z'] = zscore_col(df['num_words'])
    else:
        df['num_words_z'] = 0

    # Flesch_Kincaid
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid_z'] = zscore_col(df['Flesch_Kincaid'])
    else:
        df['Flesch_Kincaid_z'] = 0

    # Age
    if 'age' in df.columns:
        df['age_z'] = zscore_col(df['age'])
    else:
        df['age_z'] = 0

    # Keep group id and page id for modeling
    df['uuid'] = df['uuid'].astype(str)
    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype(str)
    else:
        df['page_id'] = 'unknown'

    # Final sanitation: drop rows with any remaining NA in model columns
    model_cols = [
        'speed', 'reader_view', 'dyslexia_bin', 'reader_view_x_dyslexia',
        'num_words_z', 'Flesch_Kincaid_z', 'age_z', 'retake_trial', 'english_native_bin',
        'device_smartphone', 'device_tablet', 'uuid', 'page_id'
    ]
    df = df.dropna(subset=model_cols)

    # Return the transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects linear model to estimate the effect of reader_view on speed,
    and whether that effect differs by dyslexia status.

    Model specification (primary):
    speed ~ reader_view + dyslexia_bin + reader_view_x_dyslexia
            + num_words_z + Flesch_Kincaid_z + age_z
            + retake_trial + english_native_bin
            + device_tablet + device_smartphone
    Random effects: random intercept for each participant (uuid)

    The interaction term reader_view_x_dyslexia lets us estimate the effect of reader_view specifically
    for readers with dyslexia (effect = coef(reader_view) + coef(reader_view_x_dyslexia)).
    """
    import numpy as np

    required = [
        'speed', 'reader_view', 'dyslexia_bin', 'reader_view_x_dyslexia',
        'num_words_z', 'Flesch_Kincaid_z', 'age_z', 'retake_trial', 'english_native_bin',
        'device_tablet', 'device_smartphone', 'uuid'
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column for model not found in dataframe: {c}")

    # Build formula. Keep main effects and interaction already as a column for stability.
    formula = (
        'speed ~ reader_view + dyslexia_bin + reader_view_x_dyslexia '
        '+ num_words_z + Flesch_Kincaid_z + age_z '
        '+ retake_trial + english_native_bin + device_tablet + device_smartphone'
    )

    # Fit mixed effects model with random intercepts by participant (uuid)
    md = smf.mixedlm(formula, data=df, groups=df['uuid'])
    try:
        mdf = md.fit(reml=False, method='lbfgs')
    except Exception:
        # fallback to default optimizer settings if lbfgs fails
        mdf = md.fit(reml=False)

    # Compute the estimated effect of reader_view for dyslexic participants
    # effect_dys = coef(reader_view) + coef(reader_view_x_dyslexia)
    params = mdf.params
    cov = mdf.cov_params()
    # safe extraction with zeros if parameter missing
    b_rv = float(params.get('reader_view', 0.0))
    b_int = float(params.get('reader_view_x_dyslexia', 0.0))
    effect_dys = b_rv + b_int

    # compute se(effect) using variance formula Var(a+b)=Var(a)+Var(b)+2Cov(a,b)
    se = np.nan
    try:
        v_rv = cov.loc['reader_view', 'reader_view']
        v_int = cov.loc['reader_view_x_dyslexia', 'reader_view_x_dyslexia']
        covar = cov.loc['reader_view', 'reader_view_x_dyslexia']
        se = float(np.sqrt(v_rv + v_int + 2 * covar))
    except Exception:
        se = np.nan

    # t and p-value (approx using Normal; for mixed models degrees of freedom are complex)
    if not np.isnan(se) and se > 0:
        t_stat = effect_dys / se
        # two-sided p-value using standard normal approximation
        from scipy import stats
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.nan
        p_val = np.nan

    # Prepare a short summary dict to return alongside the fitted model
    summary = {
        'model_result': mdf,
        'coef_reader_view': float(b_rv),
        'coef_interaction_reader_view_x_dyslexia': float(b_int),
        'effect_reader_view_for_dyslexic': float(effect_dys),
        'se_effect_reader_view_for_dyslexic': float(se) if not np.isnan(se) else None,
        't_effect_reader_view_for_dyslexic': float(t_stat) if not np.isnan(t_stat) else None,
        'p_effect_reader_view_for_dyslexic': float(p_val) if not np.isnan(p_val) else None,
    }

    # Print model summary for user convenience
    print(mdf.summary())
    print("Estimated effect of Reader View for dyslexic readers:")
    se_print = summary['se_effect_reader_view_for_dyslexic']
    t_print = summary['t_effect_reader_view_for_dyslexic']
    p_print = summary['p_effect_reader_view_for_dyslexic']
    print(f"Effect = {summary['effect_reader_view_for_dyslexic']:.4f}, "
          f"SE = {se_print:.4f} if SE available else None, "
          f"t = {t_print}, p ~ {p_print}")

    return summary