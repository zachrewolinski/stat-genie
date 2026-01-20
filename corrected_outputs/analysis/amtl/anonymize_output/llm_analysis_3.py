from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/anonymize_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid mutating the caller's dataframe
    df = df.copy()

    # Rename columns to meaningful names used below
    rename_map = {
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'Missing',
        'feature4': 'Sockets',
        'feature5': 'AgeAtDeath',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexEstimate',
        'feature8': 'Genus',
        'feature9': 'Region'
    }
    df = df.rename(columns=rename_map)

    # Keep only rows that have a positive number of observable sockets (trials)
    df = df[df['Sockets'].notna()]
    df = df[df['Sockets'] > 0]

    # Drop rows with missing critical variables
    df = df.dropna(subset=['Missing', 'ToothClass', 'AgeAtDeath', 'SexEstimate', 'Genus'])

    # Ensure Missing is integer and consistent with Sockets
    df['Missing'] = pd.to_numeric(df['Missing'], errors='coerce')
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce')
    # Filter out impossible values
    df = df[(df['Missing'] >= 0) & (df['Sockets'] >= 1) & (df['Missing'] <= df['Sockets'])]

    # Standardize ToothClass text values (capitalize first letter)
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip().str.capitalize()
    # If categories are spelled differently, coerce into the expected three classes
    df.loc[~df['ToothClass'].isin(['Anterior', 'Posterior', 'Premolar']), 'ToothClass'] = df['ToothClass']

    # Create binary IsHuman indicator: 1 for 'Homo sapiens', 0 for other genera (Pan, Pongo, Papio, etc.)
    df['IsHuman'] = (df['Genus'].astype(str).str.strip().str.lower() == 'homo sapiens').astype(int)

    # Create AMTL rate for descriptive checks
    df['AMTL_rate'] = df['Missing'] / df['Sockets']

    # Center age to improve model stability
    df['AgeAtDeath'] = pd.to_numeric(df['AgeAtDeath'], errors='coerce')
    age_mean = df['AgeAtDeath'].mean()
    df['AgeCentered'] = df['AgeAtDeath'] - age_mean

    # Ensure SexEstimate is numeric and within reasonable bounds; keep as-is if within [0,1], otherwise coerce to NA
    df['SexEstimate'] = pd.to_numeric(df['SexEstimate'], errors='coerce')
    df.loc[(df['SexEstimate'] < 0) | (df['SexEstimate'] > 1), 'SexEstimate'] = pd.NA

    # Final drop of any rows with newly created NA in required model columns
    df = df.dropna(subset=['Missing', 'Sockets', 'AMTL_rate', 'IsHuman', 'AgeCentered', 'SexEstimate', 'ToothClass'])

    # Keep only columns relevant for analysis (but retain some metadata)
    keep_cols = ['SpecimenID', 'Genus', 'Region', 'ToothClass', 'Missing', 'Sockets', 'AMTL_rate', 'IsHuman', 'AgeAtDeath', 'AgeCentered', 'AgeUncertainty', 'SexEstimate']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np

    # Expect df to be the output of transform(); ensure required columns are present
    required = ['Missing', 'Sockets', 'AMTL_rate', 'IsHuman', 'AgeCentered', 'SexEstimate', 'ToothClass']
    missing_req = [c for c in required if c not in df.columns]
    if len(missing_req) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing_req}")

    # Fit a binomial (logistic) GLM using counts. We model the proportion (AMTL_rate) with Sockets as weights.
    # Formula: AMTL_rate ~ IsHuman + AgeCentered + SexEstimate + C(ToothClass)
    # Using weights=Sockets ensures the binomial denominator is respected.
    formula = 'AMTL_rate ~ IsHuman + AgeCentered + SexEstimate + C(ToothClass)'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['Sockets'])
    results = model.fit()

    # Compute odds ratio and 95% CI for the IsHuman coefficient
    if 'IsHuman' in results.params.index:
        or_is_human = np.exp(results.params['IsHuman'])
        ci = results.conf_int().loc['IsHuman']
        ci_or = np.exp(ci)
    else:
        or_is_human = None
        ci_or = (None, None)

    # Pack outputs in a dict for easy inspection by calling code
    out = {
        'glm_results': results,
        'odds_ratio_IsHuman': or_is_human,
        'odds_ratio_ci_IsHuman': (float(ci_or[0]) if ci_or[0] is not None else None, float(ci_or[1]) if ci_or[1] is not None else None),
        'formula': formula
    }

    return out


