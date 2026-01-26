from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # 1) Identify which column in the raw data encodes the frequency of extramarital intercourse.
    #    The provided schema is inconsistent; the column named 'education' appears to have values 0..12 which
    #    correspond to the typical coding for frequency of affairs. Use a heuristic: choose a numeric column
    #    with min >= 0 and max <= 12 as Affair frequency if present.
    affair_candidate = None
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = pd.to_numeric(df[col], errors='coerce').min()
            col_max = pd.to_numeric(df[col], errors='coerce').max()
            if pd.notnull(col_min) and pd.notnull(col_max) and (col_min >= 0) and (col_max <= 12):
                # candidate for frequency-of-affair coding (0..12 typical)
                affair_candidate = col
                break
    # fallback: if no candidate found, but column literally named 'education' exists, use it
    if affair_candidate is None and 'education' in df.columns:
        affair_candidate = 'education'

    if affair_candidate is None:
        raise ValueError('Could not reliably detect a column encoding affair frequency. Check raw data columns.')

    # Create AffairFreq and AnyAffair
    df['AffairFreq'] = pd.to_numeric(df[affair_candidate], errors='coerce')
    # AnyAffair: 1 if AffairFreq > 0, 0 if AffairFreq == 0, missing if AffairFreq missing
    df['AnyAffair'] = (df['AffairFreq'] > 0).astype('Int64')

    # 2) Identify children indicator column. Heuristics: look for a column with yes/no values; otherwise
    #    look for a column named 'children' or 'age' (schema confusion). We will attempt to detect common text markers.
    children_col = None
    text_yesno = set(['yes','no','y','n','Yes','No','Y','N','TRUE','FALSE','True','False'])
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            uniques = set([str(x).strip() for x in df[col].dropna().unique()])
            if {'yes','no'}.issubset({u.lower() for u in uniques}):
                children_col = col
                break
    if children_col is None:
        # prefer a column literally named 'children' if present
        if 'children' in df.columns:
            children_col = 'children'
        elif 'age' in df.columns:
            # schema suggests 'age' sometimes encodes children yes/no
            # if age column contains yes/no-like values use that
            if df['age'].dropna().astype(str).str.lower().isin(['yes','no']).any():
                children_col = 'age'
    if children_col is None:
        # fallback: if a column contains only 0/1 values and name suggests children, use it
        if 'children' in df.columns and df['children'].dropna().isin([0,1]).all():
            children_col = 'children'
    if children_col is None:
        # last resort: create HasChildren from a column named 'age' if it only contains 'yes'/'no'
        pass

    # Make HasChildren column
    if children_col is not None:
        colvals = df[children_col].astype(str).str.strip()
        # standardize common encodings
        df['HasChildren'] = pd.NA
        df.loc[colvals.str.lower().isin(['yes','y','true','t']), 'HasChildren'] = 1
        df.loc[colvals.str.lower().isin(['no','n','false','f']), 'HasChildren'] = 0
        # if column is numeric 0/1
        if pd.api.types.is_numeric_dtype(df[children_col]):
            df.loc[df[children_col] == 1, 'HasChildren'] = 1
            df.loc[df[children_col] == 0, 'HasChildren'] = 0
        # coerce to integer nullable
        df['HasChildren'] = df['HasChildren'].astype('Int64')
    else:
        # If we could not detect a children column, attempt to infer from a column named 'age' or 'children'
        # If none is detected, create HasChildren as missing so downstream code will fail explicitly.
        df['HasChildren'] = pd.NA

    # 3) Controls: create standardized control columns where possible.
    # Age: the schema shows 'rating' contains age codes; convert to numeric
    if 'rating' in df.columns:
        df['Age'] = pd.to_numeric(df['rating'], errors='coerce')
    elif 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    else:
        df['Age'] = pd.NA

    # YearsMarried: prefer 'yearsmarried' column
    if 'yearsmarried' in df.columns:
        df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    else:
        df['YearsMarried'] = pd.NA

    # Education: as discussed, in this schema the column named 'affairs' appears to contain education codes (9..20)
    # So if 'affairs' values are > 8 we will map that to Education.
    if 'affairs' in df.columns:
        tmp_min = pd.to_numeric(df['affairs'], errors='coerce').min()
        if pd.notnull(tmp_min) and tmp_min >= 8:
            df['Education'] = pd.to_numeric(df['affairs'], errors='coerce')
        else:
            # otherwise try to use a column explicitly named 'education' if present and it doesn't look like affair-frequency
            if 'education' in df.columns and not (pd.to_numeric(df['education'], errors='coerce').min() >= 0 and pd.to_numeric(df['education'], errors='coerce').max() <= 12):
                df['Education'] = pd.to_numeric(df['education'], errors='coerce')
            else:
                df['Education'] = pd.NA
    else:
        df['Education'] = pd.NA

    # Religiousness - if present
    if 'religiousness' in df.columns:
        df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    else:
        df['Religiousness'] = pd.NA

    # Occupation - keep raw numeric code if present
    if 'occupation' in df.columns:
        df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    else:
        df['Occupation'] = pd.NA

    # Marital satisfaction / rownames
    if 'rownames' in df.columns:
        df['MaritalSatisfaction'] = pd.to_numeric(df['rownames'], errors='coerce')
    else:
        df['MaritalSatisfaction'] = pd.NA

    # Gender: try to detect a column containing 'male'/'female' labels. Many schemas are inconsistent; check 'children' or 'gender' columns.
    df['IsFemale'] = pd.NA
    # candidate columns to check for male/female labels
    for col in ['children','gender'] + [c for c in df.columns if c not in ['children','gender']]:
        if col not in df.columns:
            continue
        if df[col].dropna().astype(str).str.lower().isin(['male','female','m','f']).any():
            vals = df[col].astype(str).str.strip().str.lower()
            df.loc[vals.isin(['female','f']), 'IsFemale'] = 1
            df.loc[vals.isin(['male','m']), 'IsFemale'] = 0
            break
    # fallback: if a numeric gender column exists with small integer codes, try to map (this is risky but helpful if data encoded 1/2)
    if df['IsFemale'].isna().all():
        # look for a numeric column with only two unique values that might encode gender
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = pd.Series(df[col].dropna().unique())
                if unique_vals.nunique() == 2:
                    # adopt as gender indicator if plausible
                    # choose mapping so the larger mean indicates female if not sure: leave as NA if ambiguous
                    u = sorted(unique_vals.tolist())
                    df.loc[df[col] == u[0], 'IsFemale'] = 0
                    df.loc[df[col] == u[1], 'IsFemale'] = 1
                    break
    df['IsFemale'] = df['IsFemale'].astype('Int64')

    # 4) Final cleaning: drop rows without a valid AnyAffair or HasChildren (these are necessary for the main analysis)
    df_model = df.copy()
    df_model = df_model.dropna(subset=['AnyAffair', 'HasChildren'])

    # Keep only the columns we will use in modeling to make the downstream model function simple
    keep_cols = ['HasChildren', 'AnyAffair', 'AffairFreq', 'Age', 'YearsMarried', 'Education', 'Religiousness', 'IsFemale', 'Occupation', 'MaritalSatisfaction']
    for col in keep_cols:
        if col not in df_model.columns:
            df_model[col] = pd.NA

    # Return the dataframe to be used for modeling
    return df_model


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # df is expected to be the output of transform()
    results = {}
    # Prepare design matrix for controls
    controls = ['Age', 'YearsMarried', 'Education', 'Religiousness', 'IsFemale', 'Occupation', 'MaritalSatisfaction']
    # Keep only those controls that exist and have at least some non-missing data
    controls = [c for c in controls if c in df.columns]

    # Build X and drop rows with missing predictor data
    X = df[['HasChildren'] + controls].copy()
    # For statsmodels, convert Int64 nullable integers to numeric (float) with missing -> drop
    X = X.apply(pd.to_numeric, errors='coerce')
    y_logit = df['AnyAffair'].astype(float)

    dat = pd.concat([y_logit, X], axis=1).dropna()
    if dat.shape[0] == 0:
        raise ValueError('No rows with complete data for modelling after dropping missing values.')

    y = dat['AnyAffair']
    Xmat = dat.drop(columns=['AnyAffair'])
    Xmat = sm.add_constant(Xmat)

    # 1) Primary model: logistic regression for probability of any affair
    try:
        logit_model = sm.Logit(y, Xmat).fit(disp=False)
        results['logit_model'] = logit_model
    except Exception as e:
        results['logit_model_error'] = str(e)

    # 2) Robustness: model AffairFreq as a count-like outcome (negative binomial GLM)
    #    Use the same controls but only on rows with valid AffairFreq
    if 'AffairFreq' in df.columns:
        df_nb = pd.concat([df['AffairFreq'], X], axis=1).apply(pd.to_numeric, errors='coerce').dropna()
        if df_nb.shape[0] > 0:
            y_nb = df_nb['AffairFreq']
            Xnb = df_nb.drop(columns=['AffairFreq'])
            Xnb = sm.add_constant(Xnb)
            try:
                # Fit negative binomial as a flexible count model (robust to overdispersion)
                nb_model = sm.GLM(y_nb, Xnb, family=sm.families.NegativeBinomial()).fit()
                results['neg_binom_model'] = nb_model
            except Exception as e:
                results['neg_binom_model_error'] = str(e)
        else:
            results['neg_binom_model_error'] = 'No complete rows for negative binomial model.'
    else:
        results['neg_binom_model_error'] = 'AffairFreq column not present.'

    # 3) Report sample sizes and descriptive comparison
    try:
        total_n = df.shape[0]
        n_model = dat.shape[0]
        results['n_total'] = int(total_n)
        results['n_logit'] = int(n_model)
        # crude difference in means for AnyAffair by HasChildren for descriptive context
        descript = df.groupby('HasChildren')['AnyAffair'].agg(['mean','count']).to_dict()
        results['descriptive'] = descript
    except Exception:
        pass

    return results


