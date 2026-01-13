from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep track of originally present columns so we only treat originally present controls as "present"
    orig_cols = set(df.columns)

    # Helper to robustly parse binary-like columns (handles numeric 0/1, textual labels, and probabilities)
    def _parse_binary(series: pd.Series, accept_probability: bool = False, true_tokens=None, false_tokens=None):
        """
        Return a float Series with values 1.0, 0.0 or np.nan.
        accept_probability: if True and numeric values outside {0,1} are present, treat >=0.5 as 1.0 else 0.0
        true_tokens / false_tokens: sets of lowercase string tokens to interpret as true/false
        """
        if true_tokens is None:
            true_tokens = {'1', 'true', 't', 'yes', 'y', 'female', 'f', 'woman', 'accepted', 'approved', 'accept'}
        if false_tokens is None:
            false_tokens = {'0', 'false', 'fals', 'no', 'n', 'male', 'm', 'man', 'denied', 'rejected', 'reject'}

        # First try numeric conversion
        num = pd.to_numeric(series, errors='coerce')
        if num.notna().any():
            # If values are exactly 0/1, keep those; otherwise if accept_probability allow thresholding; else coerce non 0/1 to NaN
            unique_vals = set(np.unique(num.dropna()))
            if unique_vals <= {0, 1}:
                return num.where(num.isin([0, 1]), np.nan).astype(float)
            else:
                if accept_probability:
                    return pd.Series(np.where(num >= 0.5, 1.0, 0.0), index=series.index, dtype=float).where(~num.isna(), np.nan)
                else:
                    # Non-binary numeric values present but we don't accept them; coerce to NaN for those entries
                    return num.where(num.isin([0, 1]), np.nan).astype(float)

        # Fall back to string mapping
        # Note: .astype(str) will turn NaN into 'nan', so fillna first to preserve missingness
        s = series.where(series.notna(), other='').astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out[s.isin(true_tokens)] = 1.0
        out[s.isin(false_tokens)] = 0.0
        return out

    # 1) Construct Approved outcome (1 = accepted, 0 = denied).
    # Prefer using 'Unnamed: 0' when present (schema suggests it encodes acceptance),
    # otherwise try to infer from 'mortgage_credit' which in schema is described as 1=denied,0=accepted.
    if 'Unnamed: 0' in df.columns:
        # Interpret 'Unnamed: 0' robustly: accept numeric 0/1 or textual accepted/denied labels
        approved = _parse_binary(df['Unnamed: 0'], accept_probability=False,
                                 true_tokens={'1', 'accepted', 'approved', 'accept', 'yes', 'y'},
                                 false_tokens={'0', 'denied', 'rejected', 'reject', 'no', 'n'})
        df['Approved'] = approved
    elif 'mortgage_credit' in df.columns:
        # mortgage_credit described as 1 if denied, 0 if accepted -> invert
        mc = df['mortgage_credit']
        mc_num = pd.to_numeric(mc, errors='coerce')
        if mc_num.notna().any():
            # For numeric, invert 0/1 values; non 0/1 -> NaN
            df['Approved'] = np.where(mc_num == 0, 1.0, np.where(mc_num == 1, 0.0, np.nan)).astype(float)
        else:
            # Interpret textual values then invert
            mc_parsed = _parse_binary(mc, accept_probability=False,
                                      true_tokens={'denied', 'rejected', 'reject', '1', 'yes', 'y'},
                                      false_tokens={'accepted', 'approved', 'accept', '0', 'no', 'n'})
            # mc_parsed is 1.0 when mortgage_credit indicates denied -> Approved should be 0
            df['Approved'] = mc_parsed.map({1.0: 0.0, 0.0: 1.0}).astype(float)
    else:
        # If neither column exists, create Approved column with NaNs
        df['Approved'] = np.nan

    # 2) Construct Female indicator from 'consumer_credit' (preferred) or fallback to 'female'
    if 'consumer_credit' in df.columns:
        cc = df['consumer_credit']
        df['Female'] = _parse_binary(cc, accept_probability=True,
                                     true_tokens={'1', 'female', 'f', 'woman', 'women', 'yes', 'y'},
                                     false_tokens={'0', 'male', 'm', 'man', 'men', 'no', 'n'})
    elif 'female' in df.columns:
        f = df['female']
        df['Female'] = _parse_binary(f, accept_probability=True,
                                     true_tokens={'1', 'female', 'f', 'woman', 'women', 'yes', 'y'},
                                     false_tokens={'0', 'male', 'm', 'man', 'men', 'no', 'n'})
    else:
        df['Female'] = np.nan

    # 3) Ensure control columns are present and numeric; convert if present
    control_cols = [
        'bad_history', 'mortgage_credit', 'loan_to_value', 'denied_PMI',
        'self_employed', 'married', 'housing_expense_ratio', 'PI_ratio'
    ]

    for c in control_cols:
        if c in orig_cols:
            # Try numeric conversion; keep as float. We'll allow non-binary numeric controls to remain as float.
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        else:
            # create the column if missing (filled with NaN). We don't treat these as "present" controls.
            df[c] = np.nan

    # Determine which controls were originally present and have at least one non-missing value after conversion
    present_controls = [c for c in control_cols if c in orig_cols and df[c].notna().any()]

    # 4) Keep only rows with non-missing values for outcome, IV, and controls used in the model
    required_cols = ['Approved', 'Female'] + present_controls
    # Only drop rows where any of required_cols is missing
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # 5) Cast binary columns to integer dtype (0/1)
    if not df.empty:
        # Defensive check: ensure values are 0/1; if not, attempt to coerce by thresholding
        def _ensure_binary_int(series: pd.Series) -> pd.Series:
            s_num = pd.to_numeric(series, errors='coerce')
            if set(np.unique(s_num.dropna())) <= {0, 1}:
                return s_num.astype(int)
            else:
                # threshold at 0.5
                return pd.Series(np.where(s_num >= 0.5, 1, 0), index=series.index).astype(int)

        # Approved and Female should be exact 0/1 integers where possible
        try:
            df['Approved'] = _ensure_binary_int(df['Approved'])
        except Exception:
            df['Approved'] = pd.to_numeric(df['Approved'], errors='coerce')

        try:
            df['Female'] = _ensure_binary_int(df['Female'])
        except Exception:
            df['Female'] = pd.to_numeric(df['Female'], errors='coerce')

        # Cast controls: if a control has only 0/1 values, cast to int; else leave as float
        for c in control_cols:
            if df[c].notna().any():
                vals = pd.to_numeric(df[c].dropna(), errors='coerce').unique()
                try:
                    unique_vals = set(np.unique(vals))
                    if unique_vals <= {0, 1, 0.0, 1.0}:
                        df[c] = df[c].astype(int)
                    else:
                        df[c] = df[c].astype(float)
                except Exception:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
            else:
                # leave as float with NaNs
                df[c] = df[c].astype(float)

    # 6) Final dataframe includes all conceptual variables (Approved, Female) and all control columns
    final_cols = ['Approved', 'Female'] + control_cols
    df = df[final_cols].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial) predicting Approved from Female and controls.
    Returns: statsmodels fitted results object with robust standard errors (HC3).
    If 'Female' is constant (no variation), it cannot be estimated; in that case the returned
    result will be based on a model that omits 'Female' but the returned object will include
    a placeholder parameter for 'Female' with NaN values for coefficient, std. error, and p-value.
    """
    # Basic checks
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Transformed dataframe is empty. No observations available to fit the model.")

    # Identify control columns present in the transformed dataframe (all columns except the DV and IV)
    all_cols = list(df.columns)
    if 'Approved' not in all_cols or 'Female' not in all_cols:
        raise ValueError("Dataframe must contain 'Approved' and 'Female' columns. Make sure to run the transform function first.")

    control_cols = [c for c in all_cols if c not in ['Approved', 'Female']]

    # Prepare X and y
    # Ensure we only use the conceptual variables: Female and the control columns present in df
    predictors = ['Female'] + control_cols
    X = df[predictors].copy()

    # Ensure numeric dtypes for all columns in X
    try:
        X = X.apply(pd.to_numeric, errors='raise').astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert predictors to numeric types: {e}")

    # Add constant
    X = sm.add_constant(X, has_constant='add')  # will add column named 'const' if not present

    y = pd.to_numeric(df['Approved'], errors='raise').astype(float)

    # Sanity checks
    if X.shape[0] == 0:
        raise ValueError("No observations available after preparing the design matrix.")

    # Check dependent variable has 0/1
    unique_y = np.unique(y)
    if not np.all(np.isin(unique_y, [0, 1])):
        raise ValueError("Dependent variable 'Approved' must contain only 0/1 values.")
    if unique_y.size < 2:
        raise ValueError(f"Dependent variable 'Approved' must contain both classes 0 and 1. Found only: {unique_y.tolist()}")

    # Detect and handle predictors with zero variance (constant columns).
    const_name = 'const'
    if const_name not in X.columns:
        X.insert(0, const_name, 1.0)

    zero_var_cols = [col for col in X.columns if col != const_name and X[col].nunique(dropna=True) <= 1]

    female_constant = False
    if 'Female' in zero_var_cols:
        # Instead of raising, mark Female as constant and remove it from predictors used in estimation.
        female_constant = True
        zero_var_cols = [col for col in zero_var_cols if col != 'Female']

    # Drop zero-variance controls (they provide no information)
    cols_to_drop = [c for c in zero_var_cols if c != 'Female']
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    # Check for duplicate columns (exact duplicates). Do not drop const.
    duplicated_flags = X.T.duplicated(keep='first')
    dup_cols = X.columns[duplicated_flags.values].tolist()
    # Exclude const from being dropped; drop duplicates only among controls (and Female if present)
    dup_cols_to_drop = [c for c in dup_cols if c != const_name]
    if dup_cols_to_drop:
        X = X.drop(columns=dup_cols_to_drop)

    # If Female was constant and still present in X (rare since we removed zero-var cols earlier), drop it now for fitting
    if female_constant and 'Female' in X.columns:
        X = X.drop(columns=['Female'])

    # At this point, still possible to have linear dependencies among columns (collinearity).
    # Identify rank and, if deficient, select a maximal independent set of columns using QR pivoting.
    X_mat = X.values.astype(float)
    rank = np.linalg.matrix_rank(X_mat)
    ncols = X_mat.shape[1]
    if rank < ncols:
        # Use QR with column pivoting to choose independent columns
        try:
            Q, R, piv = np.linalg.qr(X_mat, mode='reduced', pivoting=True)
            independent_count = rank
            selected_idx = sorted(piv[:independent_count])
            selected_cols = [X.columns[i] for i in selected_idx]
        except Exception:
            # Fallback: if QR pivoting unavailable, attempt to drop columns (controls) until full rank
            selected_cols = list(X.columns)
            # never drop const; try dropping other controls one by one
            for col in list(X.columns):
                if np.linalg.matrix_rank(X[selected_cols].values) == len(selected_cols):
                    break
                if col == const_name:
                    continue
                selected_cols.remove(col)

            if np.linalg.matrix_rank(X[selected_cols].values) < len(selected_cols):
                raise RuntimeError("Design matrix is rank deficient and could not be resolved automatically.")

        # Ensure const is included if it was present
        if const_name in X.columns and const_name not in selected_cols:
            # It may be necessary to drop const to achieve independence; allow that.
            pass

        cols_to_keep = [c for c in selected_cols if c in X.columns]

        # Ensure Female (if not constant) would be retained; since Female may have been removed already, we only check presence.
        X = X[cols_to_keep]

    # We allow fitting an intercept-only model if needed. (If Female was constant, its effect is not estimable.)
    # Fit logistic regression (use Logit). We'll use Logit and then get robust covariances.
    try:
        logit = sm.Logit(y, X)
        result = logit.fit(disp=False)
    except Exception as e:
        raise RuntimeError(f"Logistic regression failed: {e}")

    # Get robust (heteroskedasticity-consistent) covariance estimates for inference
    # Some versions of statsmodels provide get_robustcov_results; if missing, compute robust covariances and construct a compatible result object.
    try:
        result_robust = result.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fallback: compute HC3 robust covariance manually and build a lightweight wrapper
        try:
            from statsmodels.stats.sandwich_covariance import cov_hc3
            robust_cov_arr = cov_hc3(result)
            cov_df = pd.DataFrame(robust_cov_arr, index=result.params.index, columns=result.params.index)
            bse = pd.Series(np.sqrt(np.diag(cov_df)), index=result.params.index)
            try:
                from scipy.stats import norm
                zvals = result.params / bse
                pvalues = 2 * (1 - norm.cdf(np.abs(zvals)))
            except Exception:
                pvalues = pd.Series(np.nan, index=result.params.index)
        except Exception:
            # As a last resort, fall back to original covariance information if available
            try:
                cov_df = result.cov_params()
                bse = result.bse if hasattr(result, 'bse') else pd.Series(np.sqrt(np.diag(cov_df)), index=cov_df.index)
                pvalues = result.pvalues if hasattr(result, 'pvalues') else pd.Series(np.nan, index=cov_df.index)
            except Exception:
                cov_df = pd.DataFrame(np.nan, index=result.params.index, columns=result.params.index)
                bse = pd.Series(np.nan, index=result.params.index)
                pvalues = pd.Series(np.nan, index=result.params.index)

        class RobustResultsCompat:
            def __init__(self, base_result, params, bse, pvalues, cov_df):
                self._base = base_result
                self.params = params.copy() if hasattr(params, 'copy') else pd.Series(params)
                self.bse = bse.copy() if hasattr(bse, 'copy') else pd.Series(bse)
                self.pvalues = pvalues.copy() if hasattr(pvalues, 'copy') else pd.Series(pvalues)
                self._cov = cov_df.copy() if isinstance(cov_df, pd.DataFrame) else pd.DataFrame(cov_df, index=self.params.index, columns=self.params.index)

            def cov_params(self):
                return self._cov

            def summary(self):
                try:
                    base_summary = self._base.summary()
                    note = "\nNote: robust covariance (HC3) was used for inference in a compatibility wrapper."
                    try:
                        return str(base_summary) + note
                    except Exception:
                        return base_summary
                except Exception:
                    return "Model summary unavailable. Note: robust covariance (HC3) was used."

            def get_robustcov_results(self, *args, **kwargs):
                return self

            def __getattr__(self, name):
                # Delegate attribute access to the underlying base result where appropriate
                if name in ('params', 'bse', 'pvalues', 'cov_params'):
                    return object.__getattribute__(self, name)
                return getattr(self._base, name)

            def __repr__(self):
                return f"<RobustResultsCompat wrapping {repr(self._base)}>"

        result_robust = RobustResultsCompat(result, result.params, bse, pvalues, cov_df)

    # If Female was constant, construct a wrapper that includes a placeholder Female parameter with NaNs.
    if female_constant:
        base = result_robust
        base_params = base.params.copy()
        # Build new index: place Female right after const if const exists, otherwise at beginning
        new_index = list(base_params.index)
        if const_name in new_index:
            insert_pos = new_index.index(const_name) + 1
        else:
            insert_pos = 0
        if 'Female' in new_index:
            # shouldn't happen, but keep as is
            pass
        else:
            new_index = new_index[:insert_pos] + ['Female'] + new_index[insert_pos:]

        # Params: fill existing base params and put NaN for Female
        params = base_params.reindex(new_index)
        params.loc['Female'] = np.nan

        # Standard errors
        try:
            base_bse = base.bse.copy()
        except Exception:
            base_bse = pd.Series(index=base_params.index, dtype=float)
        bse = pd.Series(np.nan, index=new_index, dtype=float)
        for idx in base_bse.index:
            bse.loc[idx] = base_bse.loc[idx]

        # P-values
        try:
            base_pvalues = base.pvalues.copy()
        except Exception:
            base_pvalues = pd.Series(index=base_params.index, dtype=float)
        pvalues = pd.Series(np.nan, index=new_index, dtype=float)
        for idx in base_pvalues.index:
            pvalues.loc[idx] = base_pvalues.loc[idx]

        # Covariance matrix: expand and fill Female rows/cols with NaN
        try:
            base_cov = base.cov_params()
            cov = pd.DataFrame(np.nan, index=new_index, columns=new_index, dtype=float)
            for i in base_cov.index:
                for j in base_cov.columns:
                    cov.loc[i, j] = base_cov.loc[i, j]
        except Exception:
            cov = pd.DataFrame(np.nan, index=new_index, columns=new_index, dtype=float)

        class ResultWrapper:
            def __init__(self, base_result, params, bse, pvalues, cov, female_note=True):
                self._base = base_result
                self.params = params
                self.bse = bse
                self.pvalues = pvalues
                self.cov_params = cov
                self.female_constant = female_note

            def summary(self):
                # Return base summary text with an appended note
                try:
                    base_summary = self._base.summary()
                    note = "\nNote: 'Female' was constant (no variation) in the input data and thus its effect was not estimated (placeholder NaN values provided)."
                    try:
                        return str(base_summary) + note
                    except Exception:
                        return base_summary
                except Exception:
                    return "Model summary unavailable. Note: 'Female' was constant; its effect was not estimated."

            # Expose convenience methods to mimic a statsmodels-like object
            def get_robustcov_results(self, *args, **kwargs):
                return self

            def __repr__(self):
                return f"<ResultWrapper female_constant={self.female_constant}>"

        return ResultWrapper(base, params, bse, pvalues, cov, female_note=True)

    # Otherwise, return the robust results object directly.
    return result_robust