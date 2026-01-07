from typing import Any
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe with clear column names and derived variables.

    Final returned dataframe will contain at least the following columns used in modeling:
      - Female (0/1)
      - Approved (0/1)
      - Black (0/1)
      - Female_Black (interaction: Female * Black)
      - LoanAmount, LoanAmount_z
      - HousingExpenseRatio, HousingExpenseRatio_z
      - DebtToIncome, DebtToIncome_z
      - LTV, LTV_z
      - SelfEmployed, Married, MortCreditScore, ConsCreditScore, BadCreditHistory, DeniedPMI
    """
    df = df.copy()

    # Mapping of conceptual final column names to possible source column names in raw data.
    # We try to find one of the candidate source names in the input df and rename it to the final name.
    candidates_map = {
        'LoanAmount': ['LoanAmount', 'feature1'],
        'Female': ['Female', 'feature2'],
        'Black': ['Black', 'feature3'],
        'HousingExpenseRatio': ['HousingExpenseRatio', 'feature4'],
        'SelfEmployed': ['SelfEmployed', 'feature5'],
        'Married': ['Married', 'feature6'],
        'MortCreditScore': ['MortCreditScore', 'feature7'],
        'ConsCreditScore': ['ConsCreditScore', 'feature8'],
        'BadCreditHistory': ['BadCreditHistory', 'feature9'],
        'DebtToIncome': ['DebtToIncome', 'feature10'],
        'LTV': ['LTV', 'feature12'],
        'DeniedPMI': ['DeniedPMI', 'feature13'],
        'Approved': ['Approved', 'feature14']
    }

    # Build rename mapping from existing columns to the required final names.
    rename_map = {}
    missing_final_cols = []
    for final_col, candidates in candidates_map.items():
        found = False
        for cand in candidates:
            if cand in df.columns:
                # If the candidate already has the desired final name, no rename required.
                if cand != final_col:
                    rename_map[cand] = final_col
                found = True
                break
        if not found:
            missing_final_cols.append((final_col, candidates))

    # Fallback: if some conceptual columns weren't found by name, attempt positional mapping
    # using the featureN indices if the dataset appears to be a positional feature file.
    if missing_final_cols:
        # explicit mapping of feature numbers to final column names (from problem spec)
        feature_number_to_final = {
            1: 'LoanAmount',
            2: 'Female',
            3: 'Black',
            4: 'HousingExpenseRatio',
            5: 'SelfEmployed',
            6: 'Married',
            7: 'MortCreditScore',
            8: 'ConsCreditScore',
            9: 'BadCreditHistory',
            10: 'DebtToIncome',
            12: 'LTV',
            13: 'DeniedPMI',
            14: 'Approved'
        }

        # If dataframe has enough columns, map by positional index: featureN -> df.columns[N-1]
        max_required_index = max(feature_number_to_final.keys())
        if df.shape[1] >= max_required_index:
            for num, final_col in feature_number_to_final.items():
                # Only map if final_col not already present
                if final_col in df.columns:
                    continue
                # get the column at position num-1
                orig_col = df.columns[num - 1]
                if orig_col != final_col:
                    # avoid overwriting an existing planned rename
                    if orig_col in rename_map and rename_map[orig_col] != final_col:
                        # skip conflicting mapping
                        continue
                    rename_map[orig_col] = final_col

            # Re-evaluate missing_final_cols after positional mapping
            missing_final_cols = []
            for final_col in candidates_map.keys():
                if final_col not in df.columns and final_col not in rename_map.values():
                    missing_final_cols.append((final_col, candidates_map[final_col]))

    if missing_final_cols:
        # Provide a clear error listing which final conceptual columns could not be mapped.
        missing_names = [f"{final} (searched: {cands})" for final, cands in missing_final_cols]
        raise KeyError("Input dataframe is missing required conceptual columns. "
                       "Could not find source columns for: " + "; ".join(missing_names))

    # Apply renaming so that the dataframe has the exact final column names required by the contract.
    if rename_map:
        df = df.rename(columns=rename_map)

    # List of final columns that must be present for modeling (conceptual variables).
    model_cols = [
        'Female',
        'Approved',
        'Black',
        # Female_Black will be created
        'LoanAmount', 'LoanAmount_z',
        'HousingExpenseRatio', 'HousingExpenseRatio_z',
        'DebtToIncome', 'DebtToIncome_z',
        'LTV', 'LTV_z',
        'SelfEmployed',
        'Married',
        'MortCreditScore',
        'ConsCreditScore',
        'BadCreditHistory',
        'DeniedPMI'
    ]

    # Convert all relevant source columns to numeric where appropriate (coerce errors to NaN).
    to_numeric_cols = [
        'Female', 'Approved', 'Black',
        'LoanAmount', 'HousingExpenseRatio', 'DebtToIncome', 'LTV',
        'SelfEmployed', 'Married', 'MortCreditScore', 'ConsCreditScore',
        'BadCreditHistory', 'DeniedPMI'
    ]
    for col in to_numeric_cols:
        # Only convert columns that exist (they should, given mapping logic).
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create interaction term Female_Black (will be NaN if either component is NaN).
    if 'Female' in df.columns and 'Black' in df.columns:
        df['Female_Black'] = df['Female'] * df['Black']
    else:
        # If either doesn't exist (should not happen), create NaN column to fail later with informative message
        df['Female_Black'] = np.nan

    # Now drop any rows with missing values in the final model columns (including _z placeholders).
    # Since _z columns do not exist yet, we dropna on the raw columns needed to compute them plus other model inputs.
    dropna_required = [
        'Female', 'Approved', 'Black', 'Female_Black',
        'LoanAmount', 'HousingExpenseRatio', 'DebtToIncome', 'LTV',
        'SelfEmployed', 'Married', 'MortCreditScore', 'ConsCreditScore',
        'BadCreditHistory', 'DeniedPMI'
    ]
    existing_dropna_required = [c for c in dropna_required if c in df.columns]
    df = df.dropna(subset=existing_dropna_required)

    # After dropping rows with missing critical inputs, cast integer indicator columns to ints (0/1).
    indicators = ['Female', 'Approved', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'DeniedPMI']
    for col in indicators:
        if col in df.columns:
            # Values should be numeric and non-missing now; round to nearest int and cast.
            # Use fillna(0) only if needed is not appropriate — we require no missing here.
            df[col] = df[col].round().astype(int)

    # Ensure scores remain numeric (they are already numeric from to_numeric step).
    if 'MortCreditScore' in df.columns:
        df['MortCreditScore'] = pd.to_numeric(df['MortCreditScore'], errors='coerce')
    if 'ConsCreditScore' in df.columns:
        df['ConsCreditScore'] = pd.to_numeric(df['ConsCreditScore'], errors='coerce')

    # Recompute Female_Black after casting to ensure integer multiplication
    if 'Female' in df.columns and 'Black' in df.columns:
        df['Female_Black'] = df['Female'] * df['Black']

    # Standardize continuous covariates used in the model (z-scores). Use sample std (ddof=1).
    continuous = ['LoanAmount', 'HousingExpenseRatio', 'DebtToIncome', 'LTV']
    for col in continuous:
        if col in df.columns:
            col_series = pd.to_numeric(df[col], errors='coerce')
            mean = col_series.mean()
            std = col_series.std(ddof=1)
            if pd.isna(std) or std == 0:
                df[col + '_z'] = 0.0
            else:
                df[col + '_z'] = (col_series - mean) / std

    # Final safety check: ensure model columns are present and have no missing values.
    final_model_cols = [
        'Female','Approved','Black','Female_Black',
        'LoanAmount_z','HousingExpenseRatio_z','DebtToIncome_z','LTV_z',
        'SelfEmployed','Married','MortCreditScore','ConsCreditScore','BadCreditHistory','DeniedPMI'
    ]
    existing_final_model_cols = [c for c in final_model_cols if c in df.columns]
    df = df.dropna(subset=existing_final_model_cols)

    # Return transformed dataframe
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (logit) predicting Approved (1 = accepted) from Female and controls.

    The model includes an interaction Female_Black to test whether the gender effect differs for Black applicants.
    Returns the fitted statsmodels Logit result object.
    """
    df = df.copy()

    # Dependent and independent columns used in the model
    if 'Approved' not in df.columns:
        raise KeyError("Transformed dataframe must contain 'Approved' column")

    y = df['Approved'].astype(float)

    X_cols = [
        'Female',
        'Black',
        'Female_Black',
        # standardized continuous controls
        'LoanAmount_z',
        'HousingExpenseRatio_z',
        'DebtToIncome_z',
        'LTV_z',
        # other controls
        'SelfEmployed',
        'Married',
        'MortCreditScore',
        'ConsCreditScore',
        'BadCreditHistory',
        'DeniedPMI'
    ]

    missing_X = [c for c in X_cols if c not in df.columns]
    if missing_X:
        raise KeyError(f"Transformed dataframe is missing required model columns: {missing_X}")

    X = df[X_cols].astype(float)
    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression using statsmodels Logit
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    return results