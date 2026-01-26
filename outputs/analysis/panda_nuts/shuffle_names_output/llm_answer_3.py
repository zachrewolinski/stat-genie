def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, test statistics, p-values, and 95% CIs
    for the predictors of interest (age_years, sex, received_help_bin) from a
    statsmodels result object (OLS/GLS/MixedLM or robust-wrapped results).

    Returns a dict:
      {
        "object": {
           "age_years": {coef, se, t_or_z, p, ci95, multiplicative_change_pct},
           "sex": {param_name, coef, se, t_or_z, p, ci95, interpretation},
           "received_help_bin": {coef, se, t_or_z, p, ci95, multiplicative_change_pct}
        },
        "description": "Brief interpretation..."
      }
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Helper to convert numpy/pandas scalars to native Python floats
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    # Helper to pretty-format floats for the description (returns string)
    def _fmt(x, fmt=".4g"):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        try:
            return format(float(x), fmt)
        except Exception:
            return str(x)

    # 1) Obtain parameter estimates (fixed effects) as a pandas Series (params_series)
    params_raw = None
    if hasattr(model_output, "params"):
        try:
            params_raw = model_output.params
        except Exception:
            params_raw = None
    if params_raw is None and hasattr(model_output, "fe_params"):
        try:
            params_raw = model_output.fe_params
        except Exception:
            params_raw = None
    if params_raw is None:
        raise ValueError("Model output does not expose params or fe_params.")

    # Coerce params_raw into a pandas Series, trying to preserve names if possible
    if isinstance(params_raw, pd.Series):
        params = params_raw.copy()
    else:
        # try to find parameter names from common attributes
        names = None
        if hasattr(model_output, "param_names"):
            try:
                names = list(model_output.param_names)
            except Exception:
                names = None
        if names is None and hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
            try:
                names = list(model_output.model.exog_names)
            except Exception:
                names = None
        try:
            if names is not None and len(names) == len(params_raw):
                params = pd.Series(list(params_raw), index=names)
            else:
                params = pd.Series(list(params_raw))
        except Exception:
            # final fallback
            params = pd.Series(params_raw)

    # 2) Obtain covariance matrix for parameters
    cov = None
    # If the result is a wrapper (e.g., sandwich estimators), try underlying results
    # but only if needed; primary attempt below:
    try:
        # cov_params can be a method or attribute
        cp = getattr(model_output, "cov_params")
        if callable(cp):
            cov = cp()
        else:
            cov = cp
    except Exception:
        cov = None

    # If still None, try normalized_cov_params
    if cov is None and hasattr(model_output, "normalized_cov_params"):
        try:
            cov = model_output.normalized_cov_params
        except Exception:
            cov = None

    if cov is None:
        # try underlying results attribute used by some wrappers
        if hasattr(model_output, "results"):
            try:
                cp = getattr(model_output.results, "cov_params", None)
                if callable(cp):
                    cov = cp()
                else:
                    cov = cp
            except Exception:
                cov = None

    if cov is None:
        raise ValueError("Could not extract covariance matrix from model output.")

    # Coerce cov to DataFrame with index matching params.index
    try:
        if isinstance(cov, pd.DataFrame):
            cov_df = cov.copy()
        else:
            # convert numpy array or other 2d structure to DataFrame
            cov_arr = np.asarray(cov)
            cov_df = pd.DataFrame(cov_arr, index=params.index, columns=params.index)
    except Exception:
        # If shape/index mismatch or other issue, attempt to coerce with numeric indices
        try:
            cov_arr = np.asarray(cov)
            n = len(params)
            cov_df = pd.DataFrame(cov_arr)
            # If cov_df shape doesn't match n, try to reshape if possible
            if cov_df.shape[0] != n or cov_df.shape[1] != n:
                cov_df = pd.DataFrame(cov_arr.reshape((n, n)))
            cov_df.index = params.index
            cov_df.columns = params.index
        except Exception as e:
            raise ValueError(f"Could not coerce covariance matrix to DataFrame: {e}")

    cov = cov_df

    # 3) Compute standard errors, t/z statistics, p-values, and 95% CIs
    se = pd.Series(np.sqrt(np.abs(np.diag(cov))), index=params.index)
    # protect against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        t_or_z = params.astype(float) / se.astype(float)

    # degrees of freedom for t distribution if available
    df_resid = None
    if hasattr(model_output, "df_resid"):
        try:
            df_resid = float(model_output.df_resid)
        except Exception:
            df_resid = None

    if df_resid is not None and np.isfinite(df_resid):
        crit = stats.t.ppf(0.975, df_resid)
        pvalues = 2 * stats.t.sf(np.abs(t_or_z.astype(float)), df_resid)
    else:
        crit = stats.norm.ppf(0.975)
        pvalues = 2 * stats.norm.sf(np.abs(t_or_z.astype(float)))

    ci_lower = params.astype(float) - crit * se.astype(float)
    ci_upper = params.astype(float) + crit * se.astype(float)

    # 4) Helper to find parameter rows for variables of interest
    param_index = [str(x) for x in list(params.index)]

    def find_param(name_fragments):
        """
        name_fragments: list of substrings (case-sensitive) any of which may appear
        in the parameter name; returns matching param name (as in params.index) or None.
        """
        for i, pname in enumerate(param_index):
            for frag in name_fragments:
                if frag in pname:
                    # return actual index value (could be non-string); use params.index[i]
                    return params.index[i]
        return None

    # For age: look for 'age_years' then 'age'
    age_name = find_param(['age_years', 'age'])
    # For received help: try several variants
    help_name = find_param(['received_help_bin', 'received_help', 'received.help', 'received_help'])
    # For sex: find first parameter name mentioning sex (case-insensitive)
    sex_name = None
    for i, pname in enumerate(param_index):
        if 'sex' in pname.lower() or 'c(sex)' in pname.lower():
            sex_name = params.index[i]
            break

    def summarize_param(pname):
        if pname is None:
            return None
        try:
            return {
                "param_name": str(pname),
                "coef": _f(params.loc[pname]),
                "se": _f(se.loc[pname]),
                "t_or_z": _f(t_or_z.loc[pname]),
                "p": _f(pvalues.loc[pname]),
                "ci95": [_f(ci_lower.loc[pname]), _f(ci_upper.loc[pname])]
            }
        except Exception:
            return None

    summary = {}
    summary['age_years'] = summarize_param(age_name)
    summary['received_help_bin'] = summarize_param(help_name)
    summary['sex'] = summarize_param(sex_name)

    # 5) Add multiplicative interpretation for age and received_help_bin (since DV is log(nuts/sec))
    # multiplicative change = exp(coef); percent change = (exp(coef)-1)*100
    def add_multiplicative_info(entry):
        if entry is None:
            return None
        coef = entry.get('coef', None)
        try:
            mult = float(np.exp(coef))
            pct = (mult - 1.0) * 100.0
            entry['multiplicative_change'] = _f(mult)
            entry['multiplicative_change_pct'] = _f(pct)
        except Exception:
            entry['multiplicative_change'] = None
            entry['multiplicative_change_pct'] = None
        return entry

    summary['age_years'] = add_multiplicative_info(summary['age_years'])
    summary['received_help_bin'] = add_multiplicative_info(summary['received_help_bin'])

    # 6) Create human-readable interpretation string
    interprets = []
    if summary['age_years'] is not None:
        coef = summary['age_years']['coef']
        p = summary['age_years']['p']
        mult = summary['age_years'].get('multiplicative_change')
        pct = summary['age_years'].get('multiplicative_change_pct')
        interprets.append(
            (f"Age (per year): coef={_fmt(coef)}, p={_fmt(p)}. "
             f"This corresponds to a multiplicative change of {_fmt(mult)} "
             f"({_fmt(pct, fmt='.2f')}% per year) in nuts/sec.")
            if coef is not None else "Age: no estimate."
        )
    else:
        interprets.append("Age (age_years) not found in model output.")

    if summary['sex'] is not None:
        pname = summary['sex']['param_name']
        coef = summary['sex']['coef']
        p = summary['sex']['p']
        level = None
        pname_str = str(pname)
        if 'T.' in pname_str:
            level = pname_str.split('T.')[-1].strip(']')
        # fallback: try bracketed level like 'sex[Male]' etc.
        if level is None and ('[' in pname_str and ']' in pname_str):
            try:
                level = pname_str.split('[')[-1].split(']')[0]
            except Exception:
                level = None
        interpret_level = f" (parameter {pname_str} compares level '{level}' to the reference)" if level else f" (parameter {pname_str})"
        interprets.append(
            f"Sex{interpret_level}: coef={_fmt(coef)}, p={_fmt(p)}. A positive coef means the named sex level has higher log(nuts/sec) than the reference."
        )
    else:
        interprets.append("Sex effect not found in model output (no sex parameter).")

    if summary['received_help_bin'] is not None:
        coef = summary['received_help_bin']['coef']
        p = summary['received_help_bin']['p']
        pct = summary['received_help_bin'].get('multiplicative_change_pct')
        interprets.append(
            f"Received help (1 vs 0): coef={_fmt(coef)}, p={_fmt(p)}. "
            f"Receiving help is associated with a {_fmt(pct, fmt='.2f')}% change in nuts/sec (multiplicative)."
        )
    else:
        interprets.append("Received_help_bin not found in model output.")

    description = " ; ".join(interprets)

    return {
        "object": summary,
        "description": description
    }