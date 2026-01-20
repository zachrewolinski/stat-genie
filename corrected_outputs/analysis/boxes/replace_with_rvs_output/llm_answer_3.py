def extract_final_answer(model_output):
    """
    Extracts culture-specific age slopes (effect of centered age on choosing the majority option)
    from a fitted statsmodels MNLogit (or similar) results object, along with standard errors, z-values,
    p-values, 95% CIs, and odds ratios (per 1-year change in centered age).

    Returns a dictionary with:
      - "object": pandas.DataFrame with one row per culture (including the reference/base culture)
                  and columns: slope, se, z, p, ci_lower, ci_upper, OR, OR_ci_lower, OR_ci_upper
      - "description": human-readable explanation of what the table means
    """
    import numpy as np
    import pandas as pd
    import math

    res = model_output

    # Desired outcome label (majority) as coded in original analysis
    maj_label = '2'

    # Extract params and bse; keep originals to allow flexible indexing
    params = getattr(res, "params", None)
    bse = getattr(res, "bse", None)
    if params is None or bse is None:
        raise ValueError("Provided model_output does not have 'params' and 'bse' attributes.")

    # Attempt to coerce params and bse into pandas objects if they aren't already
    params = pd.DataFrame(params) if not isinstance(params, (pd.DataFrame, pd.Series)) else params
    bse = pd.DataFrame(bse) if not isinstance(bse, (pd.DataFrame, pd.Series)) else bse

    # Helper: find a column label in a DataFrame that best matches desired outcome label
    def find_outcome_column(df, desired_label):
        if df is None:
            return None
        cols = list(df.columns)
        desired_str = str(desired_label)
        # direct matches
        for c in cols:
            if str(c) == desired_str:
                return c
        # try numeric equality (if desired_label is numeric or can be int)
        for c in cols:
            try:
                if float(c) == float(desired_label):
                    return c
            except Exception:
                pass
        # substring matches (e.g., '2' in 'choice_2' or '2' in '2.Age_c')
        for c in cols:
            if desired_str in str(c).split() or desired_str in str(c):
                return c
        # nothing found
        return None

    # Helper: find an index entry in params/bse that matches a variable name,
    # preferring exact matches like 'Age_c' but falling back to substring matches.
    def find_index_key(obj_index, var_name, forbid_substrings=None):
        desired = str(var_name)
        # exact match
        for idx in obj_index:
            if str(idx) == desired:
                return idx
        # prefer entries that contain var_name but not any forbid_substrings
        for idx in obj_index:
            s = str(idx)
            if desired in s:
                if forbid_substrings:
                    if any(fs in s for fs in forbid_substrings):
                        continue
                return idx
        # last resort: any that contains var_name
        for idx in obj_index:
            if desired in str(idx):
                return idx
        return None

    # Helper: find interaction index keys for Age_c by Culture (flexible matching)
    def find_interaction_keys(obj_index):
        keys = []
        for idx in obj_index:
            s = str(idx)
            if "Age_c" in s and "Culture" in s:
                keys.append(idx)
        # also support forms like 'Age_c:Culture_2' or 'Age_c_Culture_2'
        if not keys:
            for idx in obj_index:
                s = str(idx)
                if "Age_c" in s and "Culture" in s.replace(":", "_"):
                    keys.append(idx)
        return keys

    # Find outcome column in params/bse
    out_col_params = find_outcome_column(params, maj_label) if isinstance(params, pd.DataFrame) else None
    out_col_bse = find_outcome_column(bse, maj_label) if isinstance(bse, pd.DataFrame) else None

    # If we couldn't find a matching outcome column, default to a sensible fallback:
    # choose the last column (often the non-reference/second outcome) when columns are numeric indices [0, 1].
    if isinstance(params, pd.DataFrame) and out_col_params is None and len(params.columns) > 0:
        out_col_params = params.columns[-1]
    if isinstance(bse, pd.DataFrame) and out_col_bse is None and len(bse.columns) > 0:
        out_col_bse = bse.columns[-1]

    # If still None for both, raise informative error
    if (isinstance(params, pd.DataFrame) and out_col_params is None) and (isinstance(bse, pd.DataFrame) and out_col_bse is None):
        raise RuntimeError(
            f"Could not locate outcome column matching '{maj_label}' in model params/bse. "
            f"Available params columns: {list(params.columns)}. "
            f"Available bse columns: {list(bse.columns)}."
        )

    # Helper to get a scalar parameter value for a variable and outcome, handling different layouts
    def get_param_value(obj, var_key, outcome_key):
        # obj can be DataFrame or Series
        if isinstance(obj, pd.Series):
            # Series might have MultiIndex index
            if isinstance(obj.index, pd.MultiIndex):
                # try find tuple matching both var and outcome
                for idx in obj.index:
                    if (str(var_key) in str(idx)) and (str(outcome_key) in str(idx)):
                        return float(obj.loc[idx])
                # try reversed order
                for idx in obj.index:
                    if (str(var_key) in str(idx)) or (str(outcome_key) in str(idx)):
                        # return when at least var matches (best effort)
                        if str(var_key) in str(idx):
                            return float(obj.loc[idx])
                raise KeyError(f"No matching Series index entry found for var '{var_key}' and outcome '{outcome_key}'.")
            else:
                # simple Series with variable names as index and outcome embedded in names: search for var_key presence
                for idx in obj.index:
                    s = str(idx)
                    if str(var_key) in s and str(outcome_key) in s:
                        return float(obj.loc[idx])
                for idx in obj.index:
                    if str(var_key) in str(idx):
                        return float(obj.loc[idx])
                raise KeyError(f"No matching Series index entry found for var '{var_key}'.")
        elif isinstance(obj, pd.DataFrame):
            # DataFrame: try direct .loc[var, outcome] if possible
            # var_key and outcome_key may be index/column objects; try multiple strategies
            # 1. If var_key exists exactly in index
            try:
                if var_key in obj.index:
                    # If a matching outcome column exists, use it
                    if outcome_key in obj.columns:
                        return float(obj.loc[var_key, outcome_key])
                    # fallback: try to find column whose string contains outcome_key
                    for c in obj.columns:
                        if str(outcome_key) in str(c):
                            try:
                                return float(obj.loc[var_key, c])
                            except Exception:
                                continue
                    # if no column found, maybe obj is a single-column DataFrame; try to return that col
                    if obj.shape[1] == 1:
                        return float(obj.iloc[obj.index.get_loc(var_key), 0])
                    raise KeyError(f"No matching outcome column for '{outcome_key}' when var index '{var_key}' found. Available columns: {list(obj.columns)}")
            except Exception:
                # continue to other heuristics
                pass
            # 2. Index entries are composite strings/tuples; try to find an index entry that contains var_key
            for idx in obj.index:
                if str(var_key) in str(idx) and str(outcome_key) in str(idx):
                    return float(obj.loc[idx, obj.columns[0]] if obj.shape[1] == 1 else obj.loc[idx, find_outcome_column(obj, outcome_key)])
            for idx in obj.index:
                if str(var_key) in str(idx):
                    # choose matching column if possible
                    col = find_outcome_column(obj, outcome_key)
                    if col is not None:
                        return float(obj.loc[idx, col])
                    # else return first column
                    return float(obj.iloc[obj.index.get_loc(idx), 0])
            # 3. As a last resort, if obj has numeric columns and a single row, try to return that cell
            if obj.shape[0] == 1 and obj.shape[1] >= 1:
                try:
                    return float(obj.iloc[0, -1])
                except Exception:
                    pass
            raise KeyError(f"No matching index entry found for variable '{var_key}'. Available indices: {list(obj.index)}")
        else:
            raise TypeError("Unsupported params/bse object type.")

    # Locate the main Age_c parameter index key (non-interaction)
    params_index = params.index if isinstance(params, (pd.DataFrame, pd.Series)) else []
    age_key = find_index_key(params_index, "Age_c", forbid_substrings=["Culture"])
    if age_key is None:
        # try a more permissive search across params index
        age_key = find_index_key(params_index, "Age_c", forbid_substrings=None)
    if age_key is None:
        raise ValueError("Model does not contain an 'Age_c' coefficient in params.index. Check model specification and parameter naming.")

    # Identify interaction keys (Age_c by Culture)
    inter_keys = find_interaction_keys(params_index)

    # Determine which outcome column to use (prefer params, then bse)
    out_col = out_col_params if out_col_params is not None else out_col_bse

    # Extract baseline slope and se for Age_c
    try:
        slope_base = float(get_param_value(params, age_key, out_col))
    except Exception as e:
        raise RuntimeError(f"Could not extract Age_c coefficient for outcome {maj_label}: {e}")
    try:
        se_base = float(get_param_value(bse, age_key, out_col))
    except Exception:
        # if bse doesn't have separate structure, attempt to compute from covariance diagonal later
        se_base = None

    # If se_base is None, try to derive from cov_params diagonal
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    if se_base is None and cov is not None:
        # attempt to find covariance entry for Age_c/age_key and same outcome
        def find_cov_diag(cov_df, var_key, outcome_key):
            # search cov_df.index/columns for entries containing both var and outcome
            for idx in cov_df.index:
                if str(var_key) in str(idx) and str(outcome_key) in str(idx):
                    try:
                        return float(cov_df.loc[idx, idx])
                    except Exception:
                        continue
            return None
        var_cov = find_cov_diag(cov, age_key, out_col)
        if var_cov is not None:
            se_base = math.sqrt(var_cov)
    if se_base is None:
        raise RuntimeError("Could not determine standard error for Age_c (bse missing and cov_params did not provide it).")

    # Helper to get covariance between two parameter entries (var1, var2) for the same outcome
    def get_cov(var1_key, var2_key, outcome_key):
        if cov is None:
            return None
        # cov may be DataFrame with MultiIndex or flat index
        # Try to find index entries that contain both var and outcome
        idx1 = None
        idx2 = None
        for idx in cov.index:
            if str(var1_key) in str(idx) and str(outcome_key) in str(idx):
                idx1 = idx
                break
        for idx in cov.index:
            if str(var2_key) in str(idx) and str(outcome_key) in str(idx):
                idx2 = idx
                break
        # If found both, return cov.loc[idx1, idx2] if present
        if idx1 is not None and idx2 is not None:
            try:
                return float(cov.loc[idx1, idx2])
            except Exception:
                # maybe cov's columns are different; try converting columns to strings and searching
                for col in cov.columns:
                    if str(var2_key) in str(col) and str(outcome_key) in str(col):
                        try:
                            return float(cov.loc[idx1, col])
                        except Exception:
                            continue
        # If not found via index, try string-based search across both axes
        for i in cov.index:
            for j in cov.columns:
                if (str(var1_key) in str(i) and str(outcome_key) in str(i)) and (str(var2_key) in str(j) and str(outcome_key) in str(j)):
                    try:
                        return float(cov.loc[i, j])
                    except Exception:
                        continue
        return None

    # Build cultures list: Reference + cultures from interactions
    cultures = ["Reference"]
    culture_inter_map = {}
    for ik in inter_keys:
        s = str(ik)
        # try to extract culture name after 'Age_c_' or 'Culture_'
        culture_name = None
        if "Age_c_Culture_" in s:
            culture_name = s.split("Age_c_")[-1]
        elif "Age_c:" in s and "Culture" in s:
            # e.g., 'Age_c:Culture_2' or 'Age_c:Culture[2]'
            part = s.split("Age_c:")[-1]
            culture_name = part
        elif "Culture_" in s:
            culture_name = s.split("Culture_")[-1]
            # prefix with 'Culture_' to be explicit
            culture_name = "Culture_" + culture_name
        else:
            # fallback: use full interaction string
            culture_name = s
        cultures.append(culture_name)
        culture_inter_map[culture_name] = ik

    rows = []
    # Baseline row
    z_base = slope_base / se_base if se_base != 0 else np.nan
    p_base = 2 * (1 - (0.5 * (1 + math.erf(abs(z_base) / math.sqrt(2))))) if not np.isnan(z_base) else np.nan
    ci_lower_base = slope_base - 1.96 * se_base
    ci_upper_base = slope_base + 1.96 * se_base
    or_base = math.exp(slope_base)
    or_ci_lower = math.exp(ci_lower_base)
    or_ci_upper = math.exp(ci_upper_base)
    rows.append({
        "culture": "Reference",
        "slope": slope_base,
        "se": se_base,
        "z": z_base,
        "p": p_base,
        "ci_lower": ci_lower_base,
        "ci_upper": ci_upper_base,
        "OR": or_base,
        "OR_ci_lower": or_ci_lower,
        "OR_ci_upper": or_ci_upper,
        "interaction_var": None
    })

    # For each other culture, slope = Age_c + Age_c:Culture_k (interaction)
    for culture in cultures[1:]:
        inter_key = culture_inter_map.get(culture)
        if inter_key is None:
            continue
        # Obtain slope_inter and se_inter from params/bse using inter_key and out_col
        try:
            slope_inter = float(get_param_value(params, inter_key, out_col))
        except Exception:
            # skip if we cannot find interaction coefficient
            continue
        try:
            se_inter = float(get_param_value(bse, inter_key, out_col))
        except Exception:
            # attempt to get from covariance diagonal
            se_inter = None
            if cov is not None:
                try:
                    cov_diag = get_cov(inter_key, inter_key, out_col)
                    if cov_diag is not None:
                        se_inter = math.sqrt(cov_diag)
                except Exception:
                    se_inter = None
        if se_inter is None:
            # if still unknown, we'll approximate below using available info
            pass

        slope = slope_base + slope_inter

        # compute SE using covariance if available
        cov_11 = get_cov(age_key, age_key, out_col)
        cov_22 = get_cov(inter_key, inter_key, out_col)
        cov_12 = get_cov(age_key, inter_key, out_col)

        if cov_11 is not None and cov_22 is not None and cov_12 is not None:
            se_sum = math.sqrt(cov_11 + cov_22 + 2 * cov_12)
        else:
            # fallback: if individual ses known use sqrt(se_base^2 + se_inter^2)
            if se_base is not None and se_inter is not None:
                se_sum = math.sqrt(se_base ** 2 + se_inter ** 2)
            else:
                # last resort: set to NaN
                se_sum = np.nan

        z = slope / se_sum if se_sum not in (0, None) and not np.isnan(se_sum) else np.nan
        p = 2 * (1 - (0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))) if not np.isnan(z) else np.nan
        ci_lower = slope - 1.96 * se_sum if not np.isnan(se_sum) else np.nan
        ci_upper = slope + 1.96 * se_sum if not np.isnan(se_sum) else np.nan
        OR = math.exp(slope) if not np.isnan(slope) else np.nan
        OR_ci_lower = math.exp(ci_lower) if not np.isnan(ci_lower) else np.nan
        OR_ci_upper = math.exp(ci_upper) if not np.isnan(ci_upper) else np.nan

        rows.append({
            "culture": culture,
            "slope": slope,
            "se": se_sum,
            "z": z,
            "p": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "OR": OR,
            "OR_ci_lower": OR_ci_lower,
            "OR_ci_upper": OR_ci_upper,
            "interaction_var": str(inter_key)
        })

    result_df = pd.DataFrame(rows).set_index("culture")

    description = (
        "This table reports the estimated age slopes for the log-odds of choosing the majority option "
        "(choice code 2) versus the reference (unchosen, code 1), for the reference (omitted) culture "
        "and for each culture with an Age_c by Culture interaction. 'slope' is the change in log-odds "
        "per one-year increase in centered age. 'se' is the standard error of that slope (computed using "
        "the model covariance when available; otherwise approximated assuming zero covariance or using bse). "
        "'z' and 'p' are the Wald z-statistic and two-sided p-value testing whether the slope differs from zero. "
        "'ci_lower' and 'ci_upper' are the 95% confidence interval for the slope. 'OR' and its CI are the odds ratio "
        "(exp(slope)) and corresponding 95% CI: the multiplicative change in odds of choosing the majority "
        "option per one-year increase in centered age.\n\n"
        "Interpretation guidance: A positive significant slope indicates increasing reliance on the majority "
        "choice with age in that culture (higher odds per year). A negative significant slope indicates decreasing "
        "reliance with age. Non-significant slopes indicate no clear age-related change in reliance on the majority option."
    )

    return {"object": result_df, "description": description}