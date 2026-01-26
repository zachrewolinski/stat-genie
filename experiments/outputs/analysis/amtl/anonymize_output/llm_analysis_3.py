from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/anonymize_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe. 
    - Renames raw feature columns to descriptive names.
    - Drops rows with missing or invalid essential values.
    - Ensures counts are integers and Sockets > 0.
    - Computes AMTL_rate = Missing / Sockets.
    - Creates IsHuman binary indicator for genus.
    - Centers Age and Sex for modeling (Age_c, Sex_c).
    - Ensures ToothClass is categorical.

    Returns dataframe containing at minimum these columns:
      ['SpecimenID', 'Genus', 'ToothClass', 'Missing', 'Sockets', 'AMTL_rate',
       'IsHuman', 'AgeAtDeath', 'AgeUncertainty', 'SexEstimate', 'Age_c', 'Sex_c', 'Region']
    """
    df = df.copy()

    # Rename columns to meaningful names
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

    # Keep only rows with essential columns present
    essential_cols = ['Missing', 'Sockets', 'Genus', 'ToothClass', 'AgeAtDeath', 'SexEstimate']
    df = df.dropna(subset=essential_cols)

    # Ensure numeric types for counts and continuous variables
    # Coerce to numeric and drop rows that cannot be converted
    df['Missing'] = pd.to_numeric(df['Missing'], errors='coerce')
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce')
    df['AgeAtDeath'] = pd.to_numeric(df['AgeAtDeath'], errors='coerce')
    df['AgeUncertainty'] = pd.to_numeric(df['AgeUncertainty'], errors='coerce')
    df['SexEstimate'] = pd.to_numeric(df['SexEstimate'], errors='coerce')

    df = df.dropna(subset=['Missing', 'Sockets', 'AgeAtDeath', 'SexEstimate'])

    # Ensure sockets > 0 (can't model binomial with zero trials) and Missing is within [0, Sockets]
    df = df[df['Sockets'] > 0]

    # Round or coerce Missing to integer if needed; drop rows with impossible counts
    # Some datasets may already have integers; we coerce safely.
    df['Missing'] = df['Missing'].round().astype(int)
    df = df[df['Missing'] >= 0]

    # Drop rows where Missing > Sockets as inconsistent records (could also cap but we drop)
    df = df[df['Missing'] <= df['Sockets']]

    # Compute AMTL rate (proportion) for modeling as a binomial with weights=Sockets
    df['AMTL_rate'] = df['Missing'] / df['Sockets']

    # Create human binary indicator: 1 if genus is Homo sapiens (case-insensitive match), else 0
    # We check exact 'Homo sapiens' and also fallback to substring 'Homo' to be robust.
    df['Genus'] = df['Genus'].astype(str)
    df['IsHuman'] = df['Genus'].str.strip().str.lower().apply(lambda x: 1 if ('homo sapiens' in x) or (x == 'homo') or ('homo ' in x) else 0)

    # Make ToothClass categorical and standardize labels
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip()
    df['ToothClass'] = df['ToothClass'].replace({'Anterior': 'Anterior', 'Posterior': 'Posterior', 'Premolar': 'Premolar'})
    df['ToothClass'] = df['ToothClass'].astype('category')

    # Center continuous controls for interpretation
    df['Age_c'] = df['AgeAtDeath'] - df['AgeAtDeath'].mean()
    df['Sex_c'] = df['SexEstimate'] - df['SexEstimate'].mean()

    # Ensure Region and SpecimenID present as strings for later use
    if 'Region' in df.columns:
        df['Region'] = df['Region'].astype(str)
    df['SpecimenID'] = df['SpecimenID'].astype(str)

    # Return transformed dataframe (keep all columns, but derivatives are present)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) generalized linear model to test whether modern humans
    (Homo sapiens) have higher rates of antemortem tooth loss (AMTL) than non-human primates,
    controlling for age, sex, and tooth class.

    Model specification (proportion with weights):
      AMTL_rate ~ IsHuman + Age_c + Sex_c + C(ToothClass)
    with weights = Sockets and family = Binomial.

    Returns the fitted GLMResults object from statsmodels.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure the dataframe contains necessary columns
    required = ['AMTL_rate', 'Sockets', 'IsHuman', 'Age_c', 'Sex_c', 'ToothClass']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Formula: proportion outcome with weights equal to the number of trials (Sockets)
    formula = 'AMTL_rate ~ IsHuman + Age_c + Sex_c + C(ToothClass)'

    # Fit GLM with binomial family and weights=Sockets (treating AMTL_rate as proportion)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['Sockets'])
    results = model.fit()

    # Return the fitted results object (contains params, summary, conf_int, etc.)
    return results


