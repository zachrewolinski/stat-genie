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
    # Work on a copy
    df = df.copy()

    # Rename columns to meaningful names
    df = df.rename(columns={
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'Missing',
        'feature4': 'Sockets',
        'feature5': 'Age',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexProbMale',
        'feature8': 'Genus',
        'feature9': 'Region'
    })

    # Drop rows missing essential fields
    df = df.dropna(subset=['Missing', 'Sockets', 'Age', 'SexProbMale', 'Genus', 'SpecimenID', 'ToothClass'])

    # Coerce numeric types
    df['Missing'] = pd.to_numeric(df['Missing'], errors='coerce')
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['SexProbMale'] = pd.to_numeric(df['SexProbMale'], errors='coerce')

    # Remove rows with invalid sockets (must be >= 1)
    df = df[df['Sockets'] >= 1]

    # Ensure Missing is an integer between 0 and Sockets
    df['Missing'] = df['Missing'].fillna(0).astype(int)
    df['Sockets'] = df['Sockets'].astype(int)
    df['Missing'] = df[['Missing', 'Sockets']].apply(lambda row: int(max(0, min(row['Missing'], row['Sockets']))), axis=1)

    # Normalize text fields
    df['Genus'] = df['Genus'].astype(str).str.strip()
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip().str.capitalize()
    df['SpecimenID'] = df['SpecimenID'].astype(str)

    # Primary IV: IsHuman (1 if Homo sapiens, else 0)
    df['IsHuman'] = (df['Genus'].str.lower() == 'homo sapiens').astype(int)

    # Center age to aid model convergence and interpretability
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # --- Expand count-level rows into socket-level binary observations ---
    # For binomial logistic/GEE modeling we expand each row with Sockets entries where
    # the first 'Missing' entries are coded as Outcome=1 (missing) and the rest Outcome=0.
    rows = []
    for _, row in df.iterrows():
        m = int(row['Missing'])
        s = int(row['Sockets'])
        # Create outcome array: m ones and (s-m) zeros
        outcomes = np.concatenate([np.ones(m, dtype=int), np.zeros(s - m, dtype=int)])

        if outcomes.size == 0:
            continue

        rep = pd.DataFrame({
            'Outcome': outcomes,
            'IsHuman': row['IsHuman'],
            'Genus': row['Genus'],
            'Age': row['Age'],
            'Age_c': row['Age_c'],
            'SexProbMale': row['SexProbMale'],
            'ToothClass': row['ToothClass'],
            'SpecimenID': row['SpecimenID'],
            'Region': row.get('Region', np.nan),
            'Missing': row['Missing'],
            'Sockets': row['Sockets']
        })
        rows.append(rep)

    if len(rows) == 0:
        # Return an empty (but well-formed) dataframe with expected columns
        cols = ['Outcome', 'IsHuman', 'Genus', 'Age', 'Age_c', 'SexProbMale', 'ToothClass', 'SpecimenID', 'Region', 'Missing', 'Sockets']
        return pd.DataFrame(columns=cols)

    df_expanded = pd.concat(rows, ignore_index=True)

    # Cast categorical fields to appropriate types
    df_expanded['ToothClass'] = df_expanded['ToothClass'].astype('category')
    df_expanded['Genus'] = df_expanded['Genus'].astype('category')
    df_expanded['SpecimenID'] = df_expanded['SpecimenID'].astype('category')

    # Final dataframe returned for modeling
    return df_expanded


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial model (GEE) on the socket-level expanded dataframe produced by transform().
    The model estimates the effect of being human (IsHuman) on the probability a socket shows AMTL,
    while controlling for age, sex-estimate, and tooth class. Observations are clustered by SpecimenID
    using an exchangeable working correlation structure.

    Returns the fitted results object (statsmodels GEEResults) and prints a summary.
    """
    import statsmodels.api as sm

    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['Outcome', 'IsHuman', 'SexProbMale', 'Age_c', 'ToothClass', 'SpecimenID']
    missing_cols = [c for c in required if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing_cols}")

    # Make sure categorical variables are typed correctly
    df['ToothClass'] = df['ToothClass'].astype('category')
    df['SpecimenID'] = df['SpecimenID'].astype('category')

    # Specify family and correlation structure for clustering by specimen
    family = sm.families.Binomial()
    cov_struct = sm.cov_struct.Exchangeable()

    # Build and fit GEE using formula interface; include tooth-class as categorical
    # Formula: Outcome ~ IsHuman + SexProbMale + Age_c + C(ToothClass)
    model = sm.GEE.from_formula('Outcome ~ IsHuman + SexProbMale + Age_c + C(ToothClass)',
                                groups='SpecimenID',
                                data=df,
                                family=family,
                                cov_struct=cov_struct)

    result = model.fit()

    # Print summary and return result object for downstream inspection
    print(result.summary())

    return result


