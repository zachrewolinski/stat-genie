from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure numeric columns are numeric where present
    numeric_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create binary female-name indicator from provided gender_mf (0/1) if present
    # Ensure final column exists and is numeric (float), using np.nan when missing
    if 'gender_mf' in df.columns:
        # coerce to numeric then to float (so no pandas nullable Int64 remains)
        df['gender_female'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(float)
    else:
        df['gender_female'] = np.nan

    # Compute transformed dependent variables (log(1 + x)) to handle skew
    if 'alldeaths' in df.columns:
        df['log_deaths'] = np.where(df['alldeaths'].notna(), np.log1p(df['alldeaths']), np.nan)
        df['any_deaths'] = np.where(df['alldeaths'].notna(), (df['alldeaths'] > 0).astype(float), np.nan)
    else:
        df['log_deaths'] = np.nan
        df['any_deaths'] = np.nan

    if 'ndam15' in df.columns:
        df['log_damage'] = np.where(df['ndam15'].notna(), np.log1p(df['ndam15']), np.nan)
    else:
        df['log_damage'] = np.nan

    # Standardize continuous femininity measures (z-scores) for interpretability
    if 'masfem' in df.columns:
        masfem_mean = df['masfem'].mean(skipna=True)
        masfem_std = df['masfem'].std(skipna=True)
        # avoid division by zero
        if pd.notna(masfem_std) and masfem_std > 0:
            df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std
        else:
            df['masfem_z'] = np.nan
    else:
        df['masfem_z'] = np.nan

    if 'masfem_mturk' in df.columns:
        mt_mean = df['masfem_mturk'].mean(skipna=True)
        mt_std = df['masfem_mturk'].std(skipna=True)
        if pd.notna(mt_std) and mt_std > 0:
            df['masfem_mturk_z'] = (df['masfem_mturk'] - mt_mean) / mt_std
        else:
            df['masfem_mturk_z'] = np.nan
    else:
        df['masfem_mturk_z'] = np.nan

    # Center year to improve interpretability / numerical stability
    if 'year' in df.columns:
        df['year_c'] = df['year'] - df['year'].mean()
    else:
        df['year_c'] = np.nan

    # Ensure required control columns exist in numeric form (use NaN if missing)
    if 'wind' not in df.columns:
        df['wind'] = np.nan
    if 'min' not in df.columns:
        df['min'] = np.nan
    if 'category' not in df.columns:
        df['category'] = np.nan
    if 'elapsedyrs' not in df.columns:
        df['elapsedyrs'] = np.nan

    # Drop rows missing the minimal set required for the main analyses
    required_for_main = ['masfem_z', 'log_deaths', 'log_damage', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
    # Keep rows that have no missing values for these required columns
    keep_mask = (~df[required_for_main].isna()).all(axis=1)
    df = df.loc[keep_mask].reset_index(drop=True)

    # Convert final required columns to standard numpy numeric dtypes (float)
    # (statsmodels does not accept pandas nullable dtypes)
    final_float_cols = ['masfem_z', 'masfem_mturk_z', 'log_deaths', 'log_damage',
                        'wind', 'min', 'year_c', 'elapsedyrs', 'any_deaths', 'gender_female']
    for c in final_float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        else:
            # Ensure column exists even if absent
            df[c] = np.nan

    # category is an integer-like control; keep as float for modeling
    df['category'] = pd.to_numeric(df['category'], errors='coerce').astype(float)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    results = {}

    # Specify common controls
    controls = ['wind', 'min', 'category', 'year_c', 'elapsedyrs']

    # Helper to prepare data: select columns, coerce numeric, drop missing
    def _prepare(df_in: pd.DataFrame, predictors: list, outcome: str):
        cols = list(predictors) + controls + [outcome]
        # Ensure all requested cols exist
        for c in cols:
            if c not in df_in.columns:
                df_in[c] = np.nan
        data = df_in[cols].copy()
        # coerce to numeric (should already be numeric from transform), then drop na
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna()
        if data.empty:
            return None, None
        y = data[outcome].astype(float)
        X = data[predictors + controls].astype(float)
        X = sm.add_constant(X, has_constant='add')
        return X, y

    # 1) OLS: log_deaths ~ masfem_z + controls
    if 'masfem_z' in df.columns:
        X_deaths, y_deaths = _prepare(df, ['masfem_z'], 'log_deaths')
        if X_deaths is not None:
            ols_deaths = sm.OLS(y_deaths, X_deaths).fit()
            print('\nOLS: log_deaths ~ masfem_z + controls')
            print(ols_deaths.summary())
            results['ols_deaths_masfem'] = ols_deaths

    # 1b) OLS alternative: use binary female-name indicator instead of masfem_z
    if 'gender_female' in df.columns:
        X_deaths_g, y_deaths_g = _prepare(df, ['gender_female'], 'log_deaths')
        if X_deaths_g is not None:
            ols_deaths_gender = sm.OLS(y_deaths_g, X_deaths_g).fit()
            print('\nOLS: log_deaths ~ gender_female + controls')
            print(ols_deaths_gender.summary())
            results['ols_deaths_gender'] = ols_deaths_gender

    # 1c) Sensitivity: use masfem_mturk_z if available
    if 'masfem_mturk_z' in df.columns and df['masfem_mturk_z'].notna().any():
        X_deaths_mt, y_deaths_mt = _prepare(df, ['masfem_mturk_z'], 'log_deaths')
        if X_deaths_mt is not None:
            ols_deaths_mturk = sm.OLS(y_deaths_mt, X_deaths_mt).fit()
            print('\nOLS: log_deaths ~ masfem_mturk_z + controls')
            print(ols_deaths_mturk.summary())
            results['ols_deaths_mturk'] = ols_deaths_mturk

    # 2) OLS: log_damage ~ masfem_z + controls
    if 'masfem_z' in df.columns:
        X_damage, y_damage = _prepare(df, ['masfem_z'], 'log_damage')
        if X_damage is not None:
            ols_damage = sm.OLS(y_damage, X_damage).fit()
            print('\nOLS: log_damage ~ masfem_z + controls')
            print(ols_damage.summary())
            results['ols_damage_masfem'] = ols_damage

    # 2b) OLS alternative: binary female-name
    if 'gender_female' in df.columns:
        X_damage_g, y_damage_g = _prepare(df, ['gender_female'], 'log_damage')
        if X_damage_g is not None:
            ols_damage_gender = sm.OLS(y_damage_g, X_damage_g).fit()
            print('\nOLS: log_damage ~ gender_female + controls')
            print(ols_damage_gender.summary())
            results['ols_damage_gender'] = ols_damage_gender

    # 3) Logistic: any_deaths (binary) ~ masfem_z + controls
    if 'any_deaths' in df.columns:
        X_any, y_any = _prepare(df, ['masfem_z'], 'any_deaths')
        if X_any is not None:
            # Ensure y is 0/1
            y_any = y_any.astype(float)
            # If y has values other than 0/1, attempt to coerce  positive values to 1
            unique_vals = set(np.unique(y_any))
            if not unique_vals.issubset({0.0, 1.0}):
                y_any = (y_any > 0).astype(float)
            try:
                logit_model = sm.Logit(y_any, X_any).fit(disp=False)
                print('\nLogit: any_deaths ~ masfem_z + controls')
                print(logit_model.summary())
                results['logit_any_deaths_masfem'] = logit_model
            except Exception:
                # If Logit fails (e.g., perfect separation), skip storing
                pass

        if 'gender_female' in df.columns:
            X_any_g, y_any_g = _prepare(df, ['gender_female'], 'any_deaths')
            if X_any_g is not None:
                y_any_g = y_any_g.astype(float)
                if not set(np.unique(y_any_g)).issubset({0.0, 1.0}):
                    y_any_g = (y_any_g > 0).astype(float)
                try:
                    logit_model_g = sm.Logit(y_any_g, X_any_g).fit(disp=False)
                    print('\nLogit: any_deaths ~ gender_female + controls')
                    print(logit_model_g.summary())
                    results['logit_any_deaths_gender'] = logit_model_g
                except Exception:
                    pass

    return results