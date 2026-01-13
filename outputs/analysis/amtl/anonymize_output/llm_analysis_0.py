from typing import Any, Dict, List, Optional
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Produces the following REQUIRED final columns used by the model:
      - MissingCount: number of teeth missing for that tooth class (counts)
      - Trials: number of observable sockets (number of teeth that could be scored)
      - NonMissing: derived = Trials - MissingCount (helper)
      - PropMissing: derived = MissingCount / Trials (helper)
      - Age: estimated age at death (numeric)
      - Sex: numeric sex estimate
      - ToothClass: categorical (Anterior/Posterior/Premolar)
      - Genus: original genus string (used to compute IsHuman)
      - IsHuman: binary indicator for Homo sapiens (1) vs non-human (0)
      - SpecimenID: specimen identifier (string)
    """
    df = df.copy()

    # Helper to find a source column in the input dataframe from a list of candidates.
    def find_col(candidates: List[str]) -> Optional[str]:
        cols_lower = {c.lower(): c for c in df.columns}

        # 1) exact match or case-insensitive exact match
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]

        # 2) pattern match for feature-like names with numbers (e.g., feature1, feature_1)
        for cand in candidates:
            m = re.match(r'feature\W*([0-9]+)', cand, flags=re.I)
            if m:
                digit = m.group(1)
                for col in df.columns:
                    if re.search(rf'feature\W*{digit}', col, flags=re.I):
                        return col
                    if re.search(rf'\b{digit}\b', col):
                        return col

        # 3) token-based fuzzy match: split candidate into alphabetic tokens (handle camelCase)
        def tokens(s: str) -> List[str]:
            # insert spaces between camelCase boundaries then extract alpha tokens
            s2 = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
            return [t.lower() for t in re.findall(r'[A-Za-z]+', s2)]

        cand_tokens_list = [tokens(cand) for cand in candidates]

        for cand_tokens in cand_tokens_list:
            for col in df.columns:
                col_tokens = [t.lower() for t in re.findall(r'[A-Za-z]+', col)]
                # if any token matches or is substring of any column token (or vice versa), accept
                for ct in cand_tokens:
                    for colt in col_tokens:
                        if not ct or not colt:
                            continue
                        if ct == colt or ct in colt or colt in ct:
                            return col
                        # handle simple plural/singular mismatch
                        if ct.rstrip('s') == colt.rstrip('s'):
                            return col

        return None

    # Candidate source names for each required final column (including the final name itself)
    # Include common alternate column names expected in input datasets.
    source_map = {
        'MissingCount': [
            'MissingCount', 'missingcount', 'missing', 'num_amtl', 'numamtl', 'num_amtl',
            'num-amtl', 'amtl', 'num_missing', 'num missing', 'numMissing'
        ],
        'Trials': [
            'Trials', 'trials', 'SocketCount', 'socketcount', 'sockets', 'sockets_count',
            'socketscount', 'num_sockets', 'num sockets', 'observed_sockets'
        ],
        'Age': [
            'Age', 'age', 'estimated_age', 'est_age', 'age_at_death'
        ],
        'Sex': [
            'Sex', 'sex', 'prob_male', 'probmale', 'prob male', 'sex_prob', 'male_prob',
            'sex_probability'
        ],
        'ToothClass': [
            'ToothClass', 'toothclass', 'tooth_class', 'tooth class', 'tooth', 'class', 'tooth_type'
        ],
        'Genus': [
            'Genus', 'genus'
        ],
        'SpecimenID': [
            'SpecimenID', 'specimenid', 'specimen', 'specimen_id', 'specimen id', 'id', 'sample_id'
        ]
    }

    found: Dict[str, str] = {}
    missing_sources: List[str] = []
    for target, candidates in source_map.items():
        col = find_col(candidates)
        if col is None:
            missing_sources.append(target)
        else:
            found[target] = col

    if missing_sources:
        # If any absolutely required conceptual variable is missing, raise an informative error.
        # These are required by the analysis contract.
        raise KeyError(f"Input dataframe is missing required source columns for: {missing_sources}. "
                       f"Available columns: {list(df.columns)}")

    # Create final-named columns in the dataframe from the found source columns
    for target, src in found.items():
        df[target] = df[src]

    # Ensure numeric types for counts and age/sex columns
    df['MissingCount'] = pd.to_numeric(df['MissingCount'], errors='coerce')
    df['Trials'] = pd.to_numeric(df['Trials'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Sex'] = pd.to_numeric(df['Sex'], errors='coerce')

    # Remove rows with missing critical values
    required_final_cols = ['MissingCount', 'Trials', 'Age', 'Sex', 'ToothClass', 'Genus', 'SpecimenID']
    df = df.dropna(subset=required_final_cols)

    # Remove rows with non-positive or invalid trial counts
    df = df[df['Trials'] > 0]

    # Remove rows where MissingCount is greater than Trials (invalid)
    df = df[df['MissingCount'] <= df['Trials']]

    # Cast counts to integers (safely)
    # Use round for any fractional counts after coercion (shouldn't occur but be robust)
    df['Trials'] = df['Trials'].astype(float).round().astype(int)
    df['MissingCount'] = df['MissingCount'].astype(float).round().astype(int)

    # Derived helper columns
    df['NonMissing'] = df['Trials'] - df['MissingCount']
    df['PropMissing'] = df['MissingCount'] / df['Trials']

    # Create binary indicator for modern humans from Genus column
    df['Genus'] = df['Genus'].astype(str)
    df['IsHuman'] = (df['Genus'].str.strip().str.lower() == 'homo sapiens').astype(int)

    # Normalize ToothClass and set as categorical
    df['ToothClass'] = df['ToothClass'].astype(str).str.strip().str.capitalize()
    df['ToothClass'] = df['ToothClass'].astype('category')

    # Ensure SpecimenID is string (for clustering later)
    df['SpecimenID'] = df['SpecimenID'].astype(str)

    # Final drop of any rows with missing model-critical values introduced by coercion
    model_cols = ['MissingCount', 'Trials', 'IsHuman', 'Age', 'Sex', 'ToothClass', 'SpecimenID']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether modern humans have higher AMTL than non-human primates,
    controlling for age, sex, and tooth class. Clustered standard errors by specimen are computed
    to account for multiple tooth-class observations per specimen.

    Model specification (count form with Binomial family):
      MissingCount ~ IsHuman + Age + Sex + C(ToothClass)
    with endog provided as (successes, failures) per observation.

    Returns the fitted GLM object and a clustered-robust-covariance version of the results.
    """
    import patsy

    # Ensure required columns exist
    required = ['MissingCount', 'Trials', 'IsHuman', 'Age', 'Sex', 'ToothClass', 'SpecimenID', 'NonMissing']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe passed to model() is missing required columns: {missing}")

    # Drop any remaining rows with NA in columns used by the model
    df = df.dropna(subset=required)

    # Build design matrices using patsy. Use count-form binomial where endog is [successes, failures].
    formula = 'MissingCount ~ IsHuman + Age + Sex + C(ToothClass)'

    # Create design matrices: y will be MissingCount, X will be the design matrix
    y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')

    # Construct endog as Nx2 array: [successes, failures]
    endog = np.vstack([df['MissingCount'].values, df['NonMissing'].values]).T

    # Fit binomial GLM using counts (successes, failures)
    glm_model = sm.GLM(endog, X, family=sm.families.Binomial()).fit()

    # Compute clustered (by specimen) robust covariance results
    try:
        clustered_results = glm_model.get_robustcov_results(cov_type='cluster', groups=df['SpecimenID'])
    except Exception:
        clustered_results = glm_model

    # Return both the original glm fit and the clustered-results object
    return {
        'glm_fit': glm_model,
        'clustered_results': clustered_results
    }