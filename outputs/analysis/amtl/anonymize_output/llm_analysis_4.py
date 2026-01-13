from typing import Any
import pandas as pd
import numpy as np


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to a dataframe suitable for binomial regression of AMTL.

    Inputs (original columns expected or synonyms):
      - ToothClass: tooth class (Anterior/Posterior/Premolar)
      - SpecimenID: specimen id
      - Missing: number of teeth missing of given class
      - Sockets: number of observable sockets for that class
      - Age: estimated age at death
      - AgeUncertainty: age uncertainty (kept but not required for main model)
      - SexEstimate: sex estimate of specimen (continuous estimate between 0 and 1)
      - Genus: genus (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio')
      - Region: region

    The function attempts to detect common alternative/raw column names (e.g., feature1..feature9
    or common snake/camel-case variants) and renames them to the canonical names required by the
    analysis. The returned dataframe will always contain the required final columns listed in the
    analysis contract.
    """
    df = df.copy()

    # Helper: possible aliases for each canonical column name (lowercased)
    aliases = {
        'ToothClass': ['feature1', 'toothclass', 'tooth_class', 'class', 'tooth_classification',
                       'tooth', 'tooth_classification', 'tooth_class_name'],
        'SpecimenID': ['feature2', 'specimenid', 'specimen_id', 'id', 'specimen'],
        'Missing': ['feature3', 'missing', 'num_missing', 'n_missing', 'amtl', 'antemortem_missing',
                    'num_amtl', 'numamtl', 'antemortem_tooth_loss', 'n_amtl'],
        'Sockets': ['feature4', 'sockets', 'num_sockets', 'observable_sockets', 'n_sockets'],
        'Age': ['feature5', 'age', 'estimated_age', 'ageatdeath', 'age_at_death', 'age_years'],
        'AgeUncertainty': ['feature6', 'ageuncertainty', 'age_uncertainty', 'age_sd', 'stdev_age', 'age_std'],
        'SexEstimate': ['feature7', 'sexestimate', 'sex_estimate', 'sex', 'prob_male', 'p_male', 'probability_male'],
        'Genus': ['feature8', 'genus'],
        'Region': ['feature9', 'region', 'site', 'location', 'pop', 'population']
    }

    # Map existing input columns (case-insensitive) to canonical names
    col_map = {}
    lowered_cols = {c.lower().strip(): c for c in df.columns}
    for canonical, cand_aliases in aliases.items():
        # also consider canonical itself as possible present name
        candidates = [canonical] + cand_aliases
        found = None
        for cand in candidates:
            key = cand.lower().strip()
            if key in lowered_cols:
                found = lowered_cols[key]
                break
        if found is not None:
            col_map[found] = canonical

    # Apply rename for any found columns
    if col_map:
        df = df.rename(columns=col_map)

    # REQUIRED final columns for subsequent processing
    required_final = ['Missing', 'Sockets', 'Genus', 'ToothClass', 'Age', 'SexEstimate']

    # Check that at least the required columns exist after mapping; if not, raise a clear error
    missing_after_map = [c for c in required_final if c not in df.columns]
    if missing_after_map:
        raise ValueError(
            "Input dataframe is missing required columns after attempting to map common aliases. "
            f"Missing columns: {missing_after_map}. Available columns: {list(df.columns)}"
        )

    # Keep rows with the necessary data (drop rows with NA in required columns)
    df = df.dropna(subset=required_final).copy()

    # Ensure numeric types for counts and continuous covariates
    df['Missing'] = pd.to_numeric(df['Missing'], errors='coerce')
    df['Sockets'] = pd.to_numeric(df['Sockets'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['SexEstimate'] = pd.to_numeric(df['SexEstimate'], errors='coerce')

    # After coercion, drop rows that lost essential numeric values
    df = df.dropna(subset=['Missing', 'Sockets', 'Age', 'SexEstimate']).copy()

    # Remove rows with zero or negative sockets
    df = df[df['Sockets'] > 0].copy()

    # Cap Missing at Sockets and enforce non-negative
    df['Missing'] = df['Missing'].clip(lower=0)
    too_many_missing = df['Missing'] > df['Sockets']
    if too_many_missing.any():
        df.loc[too_many_missing, 'Missing'] = df.loc[too_many_missing, 'Sockets']

    # Proportion missing (for formula-based binomial with weights)
    df['PropMissing'] = df['Missing'] / df['Sockets']

    # Create primary independent variable: IsHuman (binary indicator)
    df['IsHuman'] = (df['Genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Ensure ToothClass is categorical and standardized strings
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip().replace({
        'anterior': 'Anterior', 'posterior': 'Posterior', 'premolar': 'Premolar',
        'Anterior': 'Anterior', 'Posterior': 'Posterior', 'Premolar': 'Premolar'
    })
    df['ToothClass'] = df['ToothClass'].astype('category')

    # Age and SexEstimate: create centered versions for model stability
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['SexEstimate_c'] = df['SexEstimate'] - df['SexEstimate'].mean()

    # Keep only relevant columns for subsequent modeling and return
    keep_cols = ['SpecimenID', 'Genus', 'Region', 'ToothClass', 'Missing', 'Sockets', 'PropMissing',
                 'IsHuman', 'Age', 'Age_c', 'AgeUncertainty', 'SexEstimate', 'SexEstimate_c']

    # Ensure all keep_cols are present in the returned dataframe (fill missing optional columns with NA)
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM for AMTL.

    Model specification:
      - Response: proportion of missing teeth (PropMissing) with weights = Sockets (number of trials).
      - Predictors: IsHuman (primary IV), Age (centered), SexEstimate (centered), and ToothClass (categorical).

    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns exist
    required = ['PropMissing', 'Sockets', 'IsHuman', 'Age_c', 'SexEstimate_c', 'ToothClass']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing_cols}")

    # Formula: model proportion with binomial family and weights equal to number of sockets
    formula = 'PropMissing ~ IsHuman + Age_c + SexEstimate_c + C(ToothClass)'

    # Fit GLM with Binomial family, using Sockets as the number of trials (weights)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), weights=df['Sockets'])
    results = model.fit()

    # Return the fitted results object (user can call summary() or inspect params)
    return results