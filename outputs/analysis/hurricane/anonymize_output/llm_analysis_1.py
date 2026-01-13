from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a clean dataframe containing the columns used in subsequent models.

    Expected input columns (as in provided schema):
      - feature1 .. feature14

    Output columns (kept/created):
      - StormID, Year, Name, MasFem, MTurkMasFem, FemaleName, Category,
        MinPressure, MaxWind, Deaths, Damage2013, Damage2015, YearsSince, Source,
        LogDeaths, LogDamage2015
    """
    df = df.copy()

    # Rename columns according to the mapping inferred from the schema
    rename_map = {
        'feature1': 'StormID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',        # expert coder masculinity-femininity index
        'feature5': 'MinPressure',   # minimum pressure at landfall
        'feature6': 'FemaleName',    # binary: 0 male, 1 female
        'feature7': 'Category',      # Saffir-Simpson category
        'feature8': 'Deaths',        # total number of deaths
        'feature9': 'Damage2013',    # normalized damage (2013)
        'feature10': 'YearsSince',   # number of years elapsed since hurricane
        'feature11': 'Source',       # source string/category
        'feature12': 'MTurkMasFem',  # MTurk ratings of name femininity
        'feature13': 'MaxWind',      # maximum wind speed at landfall
        'feature14': 'Damage2015'    # normalized damage (2015)
    }
    # Only rename columns that exist in the input
    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_rename:
        df = df.rename(columns=existing_rename)

    # Ensure numeric columns are numeric (coerce errors to NaN)
    num_cols = [
        'MasFem', 'MTurkMasFem', 'MinPressure', 'MaxWind',
        'Deaths', 'Damage2013', 'Damage2015', 'YearsSince',
        'Year', 'FemaleName', 'Category'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure Source and Name are strings
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('str')
    if 'Name' in df.columns:
        df['Name'] = df['Name'].astype('str')

    # Build list of required columns for basic modeling checks, only include those that exist
    required_for_model = []
    if 'MasFem' in df.columns:
        required_for_model.append('MasFem')

    # Prefer 'Deaths' (raw counts) if present; otherwise prefer 'LogDeaths' if already present.
    if 'Deaths' in df.columns:
        required_for_model.append('Deaths')
    elif 'LogDeaths' in df.columns:
        required_for_model.append('LogDeaths')

    # require at least one intensity measure if present in data
    for r in ['MaxWind', 'MinPressure', 'Category']:
        if r in df.columns:
            required_for_model.append(r)
            break

    # Only attempt to dropna on columns that actually exist in the dataframe
    subset = [c for c in required_for_model if c in df.columns]
    if subset:
        df = df.dropna(subset=subset)

    # Create the log-transformed dependent variable: LogDeaths = log(Deaths + 1)
    if 'Deaths' in df.columns:
        # ensure numeric has been coerced above
        df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
        df['LogDeaths'] = np.log(df['Deaths'].fillna(0) + 1)
    else:
        # If raw Deaths is not available but LogDeaths exists, coerce it; otherwise create as NaN column
        if 'LogDeaths' in df.columns:
            df['LogDeaths'] = pd.to_numeric(df['LogDeaths'], errors='coerce')
        else:
            df['LogDeaths'] = np.nan

    # Create log damage variable (use Damage2015 when available; fall back to Damage2013 if not)
    if 'Damage2015' in df.columns:
        df['Damage2015'] = pd.to_numeric(df['Damage2015'], errors='coerce')
        df['LogDamage2015'] = np.log(df['Damage2015'].fillna(0) + 1)
    else:
        # fallback
        df['Damage2015'] = np.nan
        if 'Damage2013' in df.columns:
            df['Damage2013'] = pd.to_numeric(df['Damage2013'], errors='coerce')
            df['LogDamage2015'] = np.log(df['Damage2013'].fillna(0) + 1)
        else:
            df['LogDamage2015'] = 0.0

    # Ensure FemaleName is 0/1 (coerce anything nonzero to 1). If missing, keep as NaN.
    if 'FemaleName' in df.columns:
        def _to_binary(x):
            if pd.isna(x):
                return np.nan
            try:
                xi = float(x)
                if xi == 1:
                    return 1.0
                if xi == 0:
                    return 0.0
            except Exception:
                pass
            s = str(x).strip()
            if s == '1':
                return 1.0
            if s == '0':
                return 0.0
            return np.nan

        df['FemaleName'] = df['FemaleName'].apply(_to_binary).astype('float')

    # Some downstream modeling uses categorical Category and Source; ensure Category is integer-like if present
    if 'Category' in df.columns:
        df['Category'] = pd.to_numeric(df['Category'], errors='coerce')
        df.loc[df['Category'].notnull(), 'Category'] = df.loc[df['Category'].notnull(), 'Category'].round().astype('Int64')

    # Keep only columns needed for modeling and inspection
    keep_cols = [
        'StormID', 'Year', 'Name', 'MasFem', 'MTurkMasFem', 'FemaleName',
        'Category', 'MinPressure', 'MaxWind', 'Deaths', 'LogDeaths',
        'Damage2013', 'Damage2015', 'LogDamage2015', 'YearsSince', 'Source'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit statistical models that test whether more-feminine hurricane names are associated with higher fatalities (a proxy for lower precaution-taking), controlling for storm severity and other confounds.

    Returns a dictionary with fitted model results for:
      - OLS on log-deaths (robust HC3 SE)
      - Negative binomial GLM on raw death counts (if Deaths is available)
      - If negative binomial fails or Deaths missing, provide informative entries

    Required columns in df: LogDeaths, MasFem, FemaleName, MaxWind, MinPressure, Category, YearsSince, LogDamage2015, Source, (optional Deaths)
    """
    results: Dict[str, Any] = {}

    # Build a base formula for controls (categorical variables are wrapped with C())
    # Only include terms if they exist in the dataframe to avoid formula errors
    control_terms = []
    for term in ['MaxWind', 'MinPressure', 'Category', 'YearsSince', 'LogDamage2015', 'Source']:
        if term in df.columns:
            if term in ['Category', 'Source']:
                control_terms.append(f"C({term})")
            else:
                control_terms.append(term)
    formula_controls = " + ".join(control_terms) if control_terms else "1"

    # Model 1: OLS on log-transformed deaths (continuous approximation; robust SE)
    ols_formula = 'LogDeaths ~ MasFem + FemaleName'
    if formula_controls:
        ols_formula = ols_formula + ' + ' + formula_controls
    try:
        ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')
        results['ols'] = ols_model
    except Exception as e:
        results['ols_error'] = str(e)

    # Model 2: Negative binomial GLM on count of deaths (only if 'Deaths' column exists)
    if 'Deaths' in df.columns and df['Deaths'].notnull().any():
        nb_formula = 'Deaths ~ MasFem + FemaleName'
        if formula_controls:
            nb_formula = nb_formula + ' + ' + formula_controls
        try:
            nb_model = smf.glm(nb_formula, data=df, family=sm.families.NegativeBinomial()).fit()
            results['neg_binomial'] = nb_model
        except Exception as e:
            # if negative binomial fails, fall back to Poisson with robust SE
            try:
                poisson_fit = smf.glm(nb_formula, data=df, family=sm.families.Poisson()).fit()
                # get robust covariance results
                poisson_robust = poisson_fit.get_robustcov_results(cov_type='HC3')
                results['poisson_fallback'] = poisson_robust
                results['neg_binomial_error'] = str(e)
            except Exception as e2:
                results['neg_binomial_error'] = f"NB error: {e}; Poisson fallback error: {e2}"
    else:
        results['neg_binomial'] = None
        results['neg_binomial_error'] = "Column 'Deaths' not present or all missing; count models skipped."

    return results