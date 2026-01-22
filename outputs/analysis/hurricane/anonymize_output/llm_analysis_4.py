from typing import Any, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe to the modeling dataframe.

    - Rename relevant columns to clear names.
    - Coerce numeric columns to numeric, drop rows missing key variables.
    - Create log-transformed deaths (LogDeaths) and standardized predictors (z-scores).
    - Center Year and ensure categorical columns are marked as such.

    Returns the transformed dataframe containing at least the columns listed in the conceptual variables.
    """
    df = df.copy()

    # Rename the raw feature columns to interpretable names used in modeling
    rename_map = {
        'feature4': 'Femininity',        # masculinity-femininity index (continuous)
        'feature6': 'FemaleNameBinary',  # binary gender indicator of the name (0 male, 1 female)
        'feature8': 'Deaths',            # total number of deaths caused by the hurricane
        'feature13': 'MaxWindSpeed',     # maximum wind speed at landfall
        'feature5': 'MinPressure',       # minimum pressure at landfall
        'feature7': 'Category',          # Saffir-Simpson category
        'feature2': 'Year',              # year the hurricane occurred
        'feature10': 'YearsSince',       # number of years elapsed since the hurricane
        'feature11': 'Source',           # data source
        'feature12': 'MTurkFemininity',  # MTurk average femininity rating (replication)
        'feature14': 'Damage2015'        # damage normalized to 2015 dollars (kept for reference)
    }

    df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric and coerce non-numeric to NaN
    numeric_cols = ['Femininity', 'FemaleNameBinary', 'Deaths', 'MaxWindSpeed', 'MinPressure',
                    'Year', 'YearsSince', 'MTurkFemininity', 'Damage2015']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are missing essential variables for the main analyses
    required_for_model = ['Femininity', 'FemaleNameBinary', 'Deaths', 'MaxWindSpeed', 'MinPressure']
    df = df.dropna(subset=required_for_model).reset_index(drop=True)

    # Enforce binary indicator as integer 0/1
    # After dropping rows with missing FemaleNameBinary, cast to int is safe
    if 'FemaleNameBinary' in df.columns:
        df['FemaleNameBinary'] = df['FemaleNameBinary'].astype(int)

    # Create transformed outcome and standardized predictors
    # Ensure Deaths are non-negative; negative values would be problematic for counts/log
    if 'Deaths' in df.columns:
        # If any negative deaths exist, coerce to NaN and drop later in modeling
        df.loc[df['Deaths'] < 0, 'Deaths'] = np.nan
        df['LogDeaths'] = np.log1p(df['Deaths'])

    # z-score standardization (population sd, ddof=0) to make coefficients comparable
    def zscore_ser(s: pd.Series) -> pd.Series:
        s_clean = s.astype(float)
        sd = s_clean.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return (s_clean - s_clean.mean()).fillna(0.0)
        return (s_clean - s_clean.mean()) / sd

    if 'Femininity' in df.columns:
        df['Femininity_z'] = zscore_ser(df['Femininity'])
    if 'MaxWindSpeed' in df.columns:
        df['MaxWindSpeed_z'] = zscore_ser(df['MaxWindSpeed'])
    if 'MinPressure' in df.columns:
        df['MinPressure_z'] = zscore_ser(df['MinPressure'])

    # Center Year to aid interpretation of intercept
    if 'Year' in df.columns:
        df['Year_centered'] = df['Year'] - df['Year'].mean()

    # Ensure categorical variables are categorical dtype (dummies will be created in modeling)
    if 'Category' in df.columns:
        df['Category'] = df['Category'].astype('category')
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('category')

    # Keep the transformed dataframe and required columns
    keep_cols = [c for c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit two complementary models to test whether more feminine hurricane names are associated
    with fewer fatalities (a proxy for fewer precautionary measures):

    1) OLS on LogDeaths (continuous, logged outcome) with robust standard errors.
    2) Negative binomial GLM on Deaths (count outcome) to model counts and account for overdispersion.

    Controls: MaxWindSpeed_z, MinPressure_z, Category (dummies), Year_centered, YearsSince, Source (dummies), MTurkFemininity.

    Returns a dictionary with fitted model objects {'ols': ols_result, 'nb': nb_result}.
    """
    df = df.copy()

    results: Dict[str, Any] = {}

    # Prepare base predictors (only keep those present in df)
    base_vars = ['Femininity_z', 'FemaleNameBinary', 'MaxWindSpeed_z', 'MinPressure_z', 'Year_centered', 'YearsSince', 'MTurkFemininity']
    present_base = [v for v in base_vars if v in df.columns]

    # Construct design matrix X (without constant) and ensure numeric
    X = df[present_base].copy() if present_base else pd.DataFrame(index=df.index)

    # Create dummies for Category and Source if present
    if 'Category' in df.columns:
        cat_dummies = pd.get_dummies(df['Category'], prefix='Category', drop_first=True, dtype=float)
        X = pd.concat([X, cat_dummies], axis=1)
    if 'Source' in df.columns:
        src_dummies = pd.get_dummies(df['Source'], prefix='Source', drop_first=True, dtype=float)
        X = pd.concat([X, src_dummies], axis=1)

    # Replace infinite values with NaN to allow proper dropping
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fit OLS on LogDeaths with robust (HC3) standard errors, if LogDeaths present
    if 'LogDeaths' in df.columns:
        df_ols = pd.concat([df['LogDeaths'], X], axis=1)
        # Drop rows with any missing values in y or X
        df_ols = df_ols.dropna()
        if df_ols.shape[0] > 0:
            y_ols = df_ols['LogDeaths'].astype(float)
            X_ols = df_ols.drop(columns='LogDeaths')
            X_ols = sm.add_constant(X_ols, has_constant='add')
            ols_model = sm.OLS(y_ols, X_ols)
            ols_res = ols_model.fit()
            # get robust covariance results (HC3)
            ols_robust = ols_res.get_robustcov_results(cov_type='HC3')
            results['ols'] = ols_robust
        else:
            results['ols'] = None
    else:
        results['ols'] = None

    # Fit Negative Binomial on Deaths (count outcome)
    if 'Deaths' in df.columns:
        df_nb = pd.concat([df['Deaths'], X], axis=1)
        df_nb = df_nb.dropna()
        # Ensure counts are non-negative integers where possible
        if df_nb.shape[0] > 0:
            y_nb = df_nb['Deaths'].astype(float)
            X_nb = df_nb.drop(columns='Deaths')
            X_nb = sm.add_constant(X_nb, has_constant='add')
            try:
                nb_model = sm.GLM(y_nb, X_nb, family=sm.families.NegativeBinomial())
                nb_res = nb_model.fit()
                results['nb'] = nb_res
            except Exception:
                # Fallback: Poisson with robust SEs if NegativeBinomial fails to converge
                pois_model = sm.GLM(y_nb, X_nb, family=sm.families.Poisson())
                pois_res = pois_model.fit()
                results['nb'] = pois_res
        else:
            results['nb'] = None
    else:
        results['nb'] = None

    return results