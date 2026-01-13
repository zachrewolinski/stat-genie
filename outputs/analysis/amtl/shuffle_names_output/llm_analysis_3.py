from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize and derive the columns needed for modeling.

    Final dataframe will contain at minimum the following columns (these exact names are used by the model function):
      - Genus (categorical)
      - Specimen (identifier, optional)
      - Num_AMTL (integer/float): number of teeth missing antemortem for the tooth class/row
      - Sockets (integer): number of observable sockets (trials)
      - Proportion_AMTL (float): Num_AMTL / Sockets
      - Age_at_death (float): estimated age at death (if available)
      - Age_at_death_centered (float): centered age (Age_at_death - mean(Age_at_death))
      - Sex_prob_male (float in 0-1): estimated male probability (if available)
      - ToothClass (categorical): anterior/posterior/premolar

    The function attempts flexible column matching but always returns the exact column names above.
    """
    df = df.copy()

    # Normalize column names for flexible matching
    lc_map = {c.lower(): c for c in df.columns}

    # Helper to pick a column by a list of candidate names (case-insensitive)
    def pick_col(candidates):
        for cand in candidates:
            if cand.lower() in lc_map:
                return lc_map[cand.lower()]
        return None

    # Preferred column names (more sensible candidate lists)
    col_genus = pick_col(['genus', 'gen', 'taxon', 'taxon_name', 'species'])
    col_num_amtl = pick_col(['num_amtl', 'num_amtl_count', 'num_missing', 'missing_teeth', 'num_missing_teeth', 'missing'])
    col_sockets = pick_col(['sockets', 'num_sockets', 'observable_sockets', 'n_sockets', 'trials'])
    col_age = pick_col(['age_at_death', 'age', 'estimated_age', 'age_estimate'])
    col_sex_prob = pick_col(['sex_prob', 'prob_male', 'probability_male', 'prob_male_est'])
    col_toothclass = pick_col(['tooth_class', 'toothclass', 'tooth_type', 'toothgroup', 'tooth_group'])
    col_specimen = pick_col(['specimen', 'specimen_id', 'specimenid', 'id', 'identifier', 'workerid'])

    # If heuristics are needed, inspect numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Heuristic for sockets: prefer a numeric column with integer-like counts typically >=1
    if col_sockets is None:
        for c in ['sockets', 'num_sockets', 'observable_sockets', 'trials', 'n_sockets']:
            if c in df.columns:
                col_sockets = c
                break
    if col_sockets is None:
        for c in num_cols:
            s = df[c].dropna()
            if s.shape[0] == 0:
                continue
            # observable sockets typically integers >= 1 and not greater than ~64
            if (s >= 1).all() and (s.max() <= 64) and (s.mean() > 1.0):
                col_sockets = c
                break

    # Heuristic for num_amtl: prefer a numeric column with values between 0 and sockets
    if col_num_amtl is None:
        for c in num_cols:
            if c == col_sockets:
                continue
            s = df[c].dropna()
            if s.shape[0] == 0:
                continue
            # missing-teeth counts often non-negative and less than typical socket counts
            if (s >= 0).all() and (s.max() <= 32) and (s.mean() <= 10):
                col_num_amtl = c
                break

    # Heuristic for age_at_death: numeric column with plausible age range
    if col_age is None:
        for c in num_cols:
            if c in [col_sockets, col_num_amtl]:
                continue
            s = df[c].dropna()
            if s.shape[0] == 0:
                continue
            # ages typically between 0 and 120 and with many unique values
            if (s >= 0).all() and (s.max() <= 120) and (s.nunique() > 8) and (s.mean() > 5):
                col_age = c
                break

    # Heuristic for sex probability: prefer 0-1 valued numeric
    if col_sex_prob is None:
        for c in num_cols:
            s = df[c].dropna()
            if s.shape[0] == 0:
                continue
            if (s >= 0).all() and (s <= 1).all():
                col_sex_prob = c
                break

    # Heuristic for tooth class: look for an obvious categorical column
    if col_toothclass is None:
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype).startswith('category'):
                sample = df[c].dropna().astype(str).head(50).str.lower().tolist()
                if any(('anterior' in x or 'posterior' in x or 'premolar' in x or 'molar' in x or 'incisor' in x) for x in sample):
                    col_toothclass = c
                    break

    # Explicitly create the final columns using the chosen source columns

    # Genus (required)
    if col_genus is None:
        # try common capitalized forms
        for c in ['Genus', 'GENUS']:
            if c in df.columns:
                col_genus = c
                break
    if col_genus is None:
        raise ValueError('Could not find a genus column. Please provide a column containing taxon names like "Homo sapiens", "Pan", "Pongo", "Papio".')
    df['Genus'] = df[col_genus].astype(str).str.strip()

    # Num_AMTL and Sockets (required)
    if col_num_amtl is None or col_sockets is None:
        raise ValueError('Could not identify both the number of AMTL (Num_AMTL) and Sockets columns. Expected numeric columns like "num_amtl" and "sockets".')

    df['Num_AMTL'] = pd.to_numeric(df[col_num_amtl], errors='coerce')
    df['Sockets'] = pd.to_numeric(df[col_sockets], errors='coerce')

    # Remove rows with missing essential values
    df = df.dropna(subset=['Genus', 'Num_AMTL', 'Sockets']).copy()

    # Ensure numeric consistency: Sockets must be positive
    df = df[df['Sockets'] > 0].copy()

    # Cap Num_AMTL between 0 and Sockets
    df.loc[df['Num_AMTL'] > df['Sockets'], 'Num_AMTL'] = df.loc[df['Num_AMTL'] > df['Sockets'], 'Sockets']
    df.loc[df['Num_AMTL'] < 0, 'Num_AMTL'] = 0

    # Compute proportion for modeling with binomial family
    df['Proportion_AMTL'] = df['Num_AMTL'] / df['Sockets']

    # Age at death (optional)
    if col_age is not None:
        df['Age_at_death'] = pd.to_numeric(df[col_age], errors='coerce')
    else:
        df['Age_at_death'] = np.nan

    # Center age for model stability
    if df['Age_at_death'].notna().any():
        age_mean = df['Age_at_death'].mean()
        df['Age_at_death_centered'] = df['Age_at_death'] - age_mean
    else:
        # keep column present per contract; fill with zeros so downstream code sees the column
        df['Age_at_death_centered'] = 0.0

    # Sex probability
    if col_sex_prob is not None:
        df['Sex_prob_male'] = pd.to_numeric(df[col_sex_prob], errors='coerce')
    else:
        df['Sex_prob_male'] = np.nan

    # Tooth class
    if col_toothclass is not None:
        df['ToothClass'] = df[col_toothclass].astype(str).str.strip()
    else:
        df['ToothClass'] = 'unknown'

    # Specimen id (optional)
    if col_specimen is not None:
        df['Specimen'] = df[col_specimen].astype(str)

    # Restrict to rows with finite proportion values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Proportion_AMTL']).copy()

    # Convert Genus and ToothClass to categorical
    df['Genus'] = df['Genus'].astype('category')
    df['ToothClass'] = df['ToothClass'].astype('category')

    # Final housekeeping: reset index
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) GLM to test whether genus (with Homo sapiens as reference) predicts AMTL frequency while controlling for age, sex, and tooth class.

    Model specification (formula) is built to include:
      - Genus (categorical, treatment coding with Homo sapiens as reference if present)
      - Age_at_death_centered (numeric; included even if constant to preserve contract)
      - Sex_prob_male (numeric; only included if any non-missing values exist)
      - ToothClass (categorical)

    We fit the model using statsmodels' GLM with Binomial family and pass the number of trials (Sockets) as freq_weights.
    The dependent variable is Proportion_AMTL.
    """
    # Ensure required columns exist in dataframe
    required = ['Genus', 'Num_AMTL', 'Sockets', 'Proportion_AMTL', 'Age_at_death_centered', 'Sex_prob_male', 'ToothClass']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Determine reference level for Genus
    gen_levels = list(pd.Categorical(df['Genus']).categories)
    if any(x for x in gen_levels if 'homo sapiens' in str(x).lower()):
        ref = [x for x in gen_levels if 'homo sapiens' in str(x).lower()][0]
    elif any(x for x in gen_levels if str(x).lower() == 'homo'):
        ref = [x for x in gen_levels if str(x).lower() == 'homo'][0]
    else:
        ref = gen_levels[0] if len(gen_levels) > 0 else None

    if ref is None:
        raise ValueError("No genus levels available to set a reference for modeling.")

    # Escape any double quotes in the reference for safe insertion into formula
    ref_escaped = str(ref).replace('"', r'\"')

    # Build formula parts. Sex_prob_male is included only if there are any non-missing values.
    formula_parts = [f'C(Genus, Treatment(reference="{ref_escaped}"))', 'Age_at_death_centered', 'C(ToothClass)']
    include_sex = df['Sex_prob_male'].notna().any()
    if include_sex:
        formula_parts.insert(-1, 'Sex_prob_male')  # keep Sex_prob_male before ToothClass for readability

    formula = 'Proportion_AMTL ~ ' + ' + '.join(formula_parts)

    # Prepare dataframe for modeling: drop rows with missing values in variables used by the formula,
    # and ensure Sockets > 0.
    vars_needed = ['Proportion_AMTL', 'Sockets'] + [p.replace('C(', '').replace(')', '') for p in formula_parts]
    # But the above replacement isn't perfect for categorical terms; instead explicitly gather predictors:
    predictors = ['Age_at_death_centered', 'Sex_prob_male', 'ToothClass', 'Genus']
    model_cols = ['Proportion_AMTL', 'Sockets'] + predictors
    # Keep only those columns that actually exist (they should per earlier check)
    model_cols = [c for c in model_cols if c in df.columns]

    df_model = df[model_cols].copy()
    # Drop rows with missing in any columns that participate in the model (Proportion_AMTL, predictors)
    df_model = df_model.dropna(subset=[c for c in model_cols if c != 'Sockets'])
    # Also ensure Sockets positive and finite
    df_model = df_model[np.isfinite(df_model['Sockets'])]
    df_model = df_model[df_model['Sockets'] > 0].copy()

    if df_model.shape[0] == 0:
        raise ValueError('No rows remain after removing missing data required for modeling. Cannot fit GLM.')

    # Fit GLM using the prepared dataframe. Use freq_weights equal to Sockets.
    model_fit = smf.glm(formula=formula, data=df_model, family=sm.families.Binomial(), freq_weights=df_model['Sockets'])
    results = model_fit.fit()

    return results