from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# If you want to run this file as a script, you can uncomment and adjust the path below.
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (Psychology Today) survey dataframe into a cleaned dataframe
    containing the dependent variable (AnyAffair), the primary IV (HasChildren), and control variables.

    Returns a dataframe that includes the following columns (always present, possibly containing NA):
    - AffairFreq, AnyAffair, HasChildren, Gender, Age, EducationLevel, YearsMarried,
      Religiousness, Occupation, MaritalRating
    """
    df = df.copy()

    # Ensure expected columns exist in df for safe assignment below
    # We'll create columns with NA defaults if they don't exist, then attempt to fill from raw columns
    required_cols = [
        'AffairFreq', 'AnyAffair', 'HasChildren', 'Gender', 'Age', 'EducationLevel',
        'YearsMarried', 'Religiousness', 'Occupation', 'MaritalRating'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Affair frequency stored in 'education' per schema description
    if 'education' in df.columns:
        df['AffairFreq'] = pd.to_numeric(df['education'], errors='coerce')
    else:
        df['AffairFreq'] = pd.NA

    # Binary dependent variable: any affair vs none (derived from AffairFreq)
    df['AnyAffair'] = (pd.to_numeric(df['AffairFreq'], errors='coerce') > 0).astype('Int64')

    # HasChildren: original 'age' column encodes presence of children as 'yes'/'no'
    if 'age' in df.columns:
        # normalize strings and map
        mapped = df['age'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
        # If mapping yields NaN, try numeric coercion (in case values are 0/1)
        numeric_fallback = pd.to_numeric(df['age'], errors='coerce')
        df['HasChildren'] = pd.Series(np.where(mapped.notna(), mapped, numeric_fallback), index=df.index)
        df['HasChildren'] = pd.to_numeric(df['HasChildren'], errors='coerce').astype('Int64')
    else:
        df['HasChildren'] = pd.NA

    # Gender: original 'children' column contains 'male'/'female' in this schema
    if 'children' in df.columns:
        mapped = df['children'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
        numeric_fallback = pd.to_numeric(df['children'], errors='coerce')
        df['Gender'] = pd.Series(np.where(mapped.notna(), mapped, numeric_fallback), index=df.index)
        df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce').astype('Int64')
    else:
        df['Gender'] = pd.NA

    # Age (numeric-coded midpoints) from 'rating'
    if 'rating' in df.columns:
        df['Age'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['Age'] = pd.NA

    # EducationLevel: from original 'affairs' column per provided mapping
    if 'affairs' in df.columns:
        df['EducationLevel'] = pd.to_numeric(df['affairs'], errors='coerce')
    else:
        df['EducationLevel'] = pd.NA

    # YearsMarried: prefer explicit 'yearsmarried' column; fall back to mislabelled 'gender' if needed
    if 'yearsmarried' in df.columns:
        df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    else:
        df['YearsMarried'] = pd.NA

    if (df['YearsMarried'].isna()).all() and 'gender' in df.columns:
        # Some schemas mislabel yearsmarried as 'gender'
        df['YearsMarried'] = pd.to_numeric(df['gender'], errors='coerce')

    # Religiousness, Occupation, MaritalRating
    if 'religiousness' in df.columns:
        df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    else:
        df['Religiousness'] = pd.NA

    if 'occupation' in df.columns:
        df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    else:
        df['Occupation'] = pd.NA

    if 'rownames' in df.columns:
        df['MaritalRating'] = pd.to_numeric(df['rownames'], errors='coerce')
    else:
        df['MaritalRating'] = pd.NA

    # Final cleaning: drop rows missing DV or primary IV (AnyAffair and HasChildren are required)
    # Coerce AnyAffair and HasChildren to numeric before drop to ensure proper NA detection
    df['AnyAffair'] = pd.to_numeric(df['AnyAffair'], errors='coerce')
    df['HasChildren'] = pd.to_numeric(df['HasChildren'], errors='coerce')
    df = df.dropna(subset=['AnyAffair', 'HasChildren'])

    # Ensure AnyAffair is integer 0/1 where possible
    df['AnyAffair'] = df['AnyAffair'].astype(int)

    # Ensure the final dataframe contains all required columns (in the specified order)
    final_cols = required_cols
    for col in final_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the binary outcome AnyAffair from HasChildren
    controlling for gender, age, education level, years married, religiousness, occupation, and marital rating.

    Returns the fitted statsmodels result object. Also prints an odds-ratio table for interpretation.
    """
    import numpy as np

    # Copy input
    df = df.copy()

    # Define predictors list in the transformed dataframe (must match conceptual variables)
    predictors = ['HasChildren', 'Gender', 'Age', 'EducationLevel', 'YearsMarried', 'Religiousness', 'Occupation', 'MaritalRating']
    # Keep only predictors that exist in the dataframe
    predictors = [p for p in predictors if p in df.columns]

    if 'AnyAffair' not in df.columns:
        raise ValueError("Input dataframe must contain 'AnyAffair' column produced by transform().")

    # Take subset and coerce predictors and outcome to numeric types suitable for statsmodels
    subset_cols = ['AnyAffair'] + predictors
    model_df = df.loc[:, subset_cols].copy()

    # Coerce all to numeric (this will convert pandas extension dtypes to numpy float)
    for col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    # Drop any rows with missing data after coercion
    model_df = model_df.dropna(axis=0, how='any')

    if model_df.shape[0] < 30:
        # Warn if sample very small; still attempt to fit
        print(f"Warning: small sample size after dropping NAs: {model_df.shape[0]} rows")

    if model_df.shape[0] == 0:
        raise ValueError("No data available after dropping rows with missing predictors/outcome.")

    # Prepare design matrices
    X = model_df[predictors]
    X = sm.add_constant(X, has_constant='add')
    # Ensure X is a plain numpy numeric array backing (float)
    X = X.astype(float)
    y = model_df['AnyAffair'].astype(float)

    # Fit logistic regression (binomial logit)
    logit_model = sm.Logit(y, X)
    try:
        results = logit_model.fit(disp=False)
    except Exception as e:
        # If convergence or perfect separation issues occur, try fallback to GLM with binomial family
        print('Logit failed to converge or had an error:', e)
        glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
        results = glm_binom.fit()

    # Prepare odds ratios and 95% CI
    try:
        params = results.params
        conf = results.conf_int()
        odds_ratios = np.exp(params)
        conf_odds = np.exp(conf)
        or_table = (odds_ratios.to_frame(name='OR')
                    .join(conf_odds.rename(columns={0: '2.5%', 1: '97.5%'})))
        # Print a concise summary table
        print('\nLogistic regression results (odds ratios with 95% CI):')
        print(or_table)
    except Exception:
        # If results structure unexpected, skip OR printing
        pass

    # Return the fitted results object for further inspection
    return results