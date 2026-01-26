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

    # --- 1) Create/clean dependent variable: affairs_count ---
    # Coerce to numeric; typical coding in this dataset uses 0 for none and higher integers for more frequent affairs.
    df['affairs_count'] = pd.to_numeric(df.get('affairs'), errors='coerce')

    # --- 2) Independent variable: HasChildren ---
    # There is ambiguity in schema; several columns may carry children/gender info. We try a robust inference.
    def detect_has_children(row):
        # priority 1: explicit column named 'children'
        v = row.get('children', None)
        if pd.notna(v):
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ('yes', 'y', '1', 'true', 't'):
                    return 1
                if s in ('no', 'n', '0', 'false', 'f'):
                    return 0
                # sometimes 'children' column is actually gender (male/female) in broken schema; skip
            # numeric coded children indicator (0/1)
            try:
                iv = int(float(v))
                if iv in (0, 1):
                    return iv
            except Exception:
                pass
        # priority 2: some versions have 'age' column as factor indicating children yes/no (per provided schema)
        v2 = row.get('age', None)
        if pd.notna(v2):
            if isinstance(v2, str):
                s = v2.strip().lower()
                if s in ('yes', 'y'):
                    return 1
                if s in ('no', 'n'):
                    return 0
            try:
                iv2 = int(float(v2))
                if iv2 in (0, 1):
                    return iv2
            except Exception:
                pass
        # If nothing matched, return NaN (will be dropped later)
        return np.nan

    df['HasChildren'] = df.apply(detect_has_children, axis=1)

    # --- 3) Controls: create/standardize columns ---
    # Education
    df['Education'] = pd.to_numeric(df.get('education'), errors='coerce')

    # Religiousness
    df['Religiousness'] = pd.to_numeric(df.get('religiousness'), errors='coerce')

    # MarriageRating: 'rownames' in schema corresponded to self-rating of marriage
    df['MarriageRating'] = pd.to_numeric(df.get('rownames'), errors='coerce')

    # Age: many schemas use 'rating' to encode age bracket (17.5,...,57)
    # We'll prefer 'rating' if it looks like ages (min>15), otherwise fallback to 'age' if numeric.
    candidate_age = pd.to_numeric(df.get('rating'), errors='coerce')
    if candidate_age.notna().sum() > 0 and candidate_age.min(skipna=True) >= 15:
        df['Age'] = candidate_age
    else:
        df['Age'] = pd.to_numeric(df.get('age'), errors='coerce')

    # YearsMarried: prefer explicit 'yearsmarried' column, otherwise try 'gender' if that looks numeric-coded for years married
    df['YearsMarried'] = pd.to_numeric(df.get('yearsmarried'), errors='coerce')
    if df['YearsMarried'].isna().sum() == len(df):
        df['YearsMarried'] = pd.to_numeric(df.get('gender'), errors='coerce')

    # Occupation
    df['Occupation'] = pd.to_numeric(df.get('occupation'), errors='coerce')

    # IsFemale: try to infer from any column that looks like gender strings or codes
    def detect_female(row):
        # check 'gender' column first (if textual)
        g = row.get('gender', None)
        if pd.notna(g):
            if isinstance(g, str):
                s = g.strip().lower()
                if s in ('female', 'f', 'woman', 'woman '):
                    return 1
                if s in ('male', 'm', 'man'):
                    return 0
            # if numeric and clearly 1/2 codes appear (common coding: 1=male, 2=female or vice versa) handle heuristically
            try:
                gv = int(float(g))
                if gv in (0, 1):
                    # assume 1 = female if distribution/labels unknown (but better to check unique values)
                    return gv
                if gv in (1, 2):
                    # guess 2->female (common), map 2->1
                    return 1 if gv == 2 else 0
            except Exception:
                pass
        # fallback: sometimes 'children' column was mislabelled and contains 'male'/'female'
        c = row.get('children', None)
        if isinstance(c, str):
            s = c.strip().lower()
            if s in ('female', 'f', 'woman'):
                return 1
            if s in ('male', 'm', 'man'):
                return 0
        return np.nan

    df['IsFemale'] = df.apply(detect_female, axis=1)

    # --- 4) Final cleaning ---
    # Drop rows missing the core variables (affairs_count or HasChildren)
    df = df.dropna(subset=['affairs_count', 'HasChildren'])

    # Convert HasChildren and IsFemale to integer types where possible
    df['HasChildren'] = df['HasChildren'].astype(int)
    if df['IsFemale'].notna().any():
        # where IsFemale is not null, cast to integer; else fill with mode or 0 if completely missing
        if df['IsFemale'].isna().all():
            df['IsFemale'] = 0
        else:
            # fill missing IsFemale with modal value to avoid dropping many rows (alternatively could drop)
            df['IsFemale'] = df['IsFemale'].fillna(df['IsFemale'].mode().iloc[0]).astype(int)

    # Rename any remaining columns used for interpretability are already set
    # Keep only columns needed for modeling to make downstream modeling code simple
    final_cols = ['affairs_count', 'HasChildren', 'Age', 'YearsMarried', 'Education',
                  'Religiousness', 'MarriageRating', 'IsFemale', 'Occupation']
    # If some of these columns do not exist, they'll be created with NaN and later handled by the model function
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # This function assumes df is the transformed dataframe returned by transform().
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Work on a copy
    dfm = df.copy()

    # Drop rows with missing values in the variables used for modeling
    model_vars = ['affairs_count', 'HasChildren', 'Age', 'YearsMarried', 'Education',
                  'Religiousness', 'MarriageRating', 'IsFemale', 'Occupation']
    dfm = dfm.dropna(subset=['affairs_count', 'HasChildren'])

    # For control variables, we'll fill modest amounts of missing data with median (alternatively drop if many missing).
    for v in ['Age', 'YearsMarried', 'Education', 'Religiousness', 'MarriageRating', 'IsFemale', 'Occupation']:
        if v in dfm.columns:
            if dfm[v].isna().sum() > 0:
                # fill with median for numeric (IsFemale will be integer or modal)
                if dfm[v].dtype.kind in 'biufc':
                    dfm[v] = dfm[v].fillna(dfm[v].median())
                else:
                    dfm[v] = dfm[v].fillna(dfm[v].mode().iloc[0])

    # Define endogenous and exogenous variables
    endog = dfm['affairs_count']
    exog_vars = ['HasChildren', 'Age', 'YearsMarried', 'Education', 'Religiousness', 'MarriageRating', 'IsFemale', 'Occupation']
    exog = sm.add_constant(dfm[exog_vars].astype(float))

    # Fit a Zero-Inflated Negative Binomial (ZINB) model to account for excess zeros and overdispersion
    try:
        zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog, inflation='logit')
        res = zinb.fit(disp=0, maxiter=100)
    except Exception as e:
        # If ZINB fails to converge or is unavailable, fallback to a simple Negative Binomial (NB)
        try:
            nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial())
            res = nb.fit()
        except Exception as e2:
            # Final fallback: OLS on the raw outcome (not ideal for count data, but provides a baseline)
            ols = sm.OLS(endog, exog)
            res = ols.fit()

    # Return the fitted model result object. Caller can use res.summary() or inspect coefficients.
    return res


