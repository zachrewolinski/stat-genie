def extract_final_answer(model_output):
    """
    Extracts the estimated effect of ReaderView on log_reading_speed for:
      - participants without dyslexia (main ReaderView effect)
      - participants with dyslexia (main ReaderView effect + ReaderView:Dyslexia interaction)

    Returns:
      {
        "object": {
          "coef_non_dyslexic": ...,
          "se_non_dyslexic": ...,
          "t_non_dyslexic": ...,
          "p_non_dyslexic": ...,
          "ci_non_dyslexic": (lower, upper),
          "coef_dyslexic": ...,
          "se_dyslexic": ...,
          "t_dyslexic": ...,
          "p_dyslexic": ...,
          "ci_dyslexic": (lower, upper),
          "interaction_coef": ...,
          "interaction_se": ...,
          "interaction_t": ...,
          "interaction_p": ...,
          "pct_change_dyslexic": ...  # (exp(coef) - 1) * 100
        },
        "description": "... human-readable interpretation ..."
      }
    """
    import numpy as np
    from scipy import stats
    import pandas as pd

    res = model_output

    # Obtain parameter values as numpy array
    params_raw = getattr(res, "params", None)
    if params_raw is None:
        raise AttributeError("model_output has no attribute 'params'")
    params_arr = np.asarray(params_raw)

    # Obtain bse and pvalues as arrays (may be missing)
    bse_raw = getattr(res, "bse", None)
    bse_arr = np.asarray(bse_raw) if bse_raw is not None else np.full_like(params_arr, np.nan, dtype=float)

    pvalues_raw = getattr(res, "pvalues", None)
    pvalues_arr = np.asarray(pvalues_raw) if pvalues_raw is not None else np.full_like(params_arr, np.nan, dtype=float)

    # Covariance matrix
    cov = getattr(res, "cov_params", None)
    if callable(cov):
        cov_raw = cov()
    else:
        cov_raw = getattr(res, "cov", None) or getattr(res, "cov_params", None)
        if callable(cov_raw):
            cov_raw = cov_raw()
    if cov_raw is None:
        raise AttributeError("Could not retrieve covariance matrix from model_output (cov_params).")

    # Degrees of freedom residual if available
    df_resid = getattr(res, "df_resid", None)

    # Determine parameter names
    names = None
    # 1) If params has an index (e.g., pandas Series)
    if hasattr(params_raw, "index"):
        try:
            names = list(params_raw.index)
        except Exception:
            names = None
    # 2) statsmodels stores names in res.model.exog_names
    if names is None and hasattr(res, "model") and hasattr(res.model, "exog_names"):
        try:
            names = list(res.model.exog_names)
        except Exception:
            names = None
    # 3) res may have param_names or names attribute
    if names is None:
        if hasattr(res, "param_names"):
            try:
                names = list(res.param_names)
            except Exception:
                names = None
    if names is None and hasattr(res, "names"):
        try:
            names = list(res.names)
        except Exception:
            names = None
    # 4) fallback: create generic names
    if names is None:
        names = [f"param_{i}" for i in range(len(params_arr))]

    # Build index map from name to position
    index_map = {n: i for i, n in enumerate(names)}

    # Helper to get value by parameter name
    def get_val(arr, name):
        if name in index_map:
            return float(arr[index_map[name]])
        # try exact match ignoring dtype wrapper like 'ReaderView[T.1]' vs 'ReaderView'
        for n in names:
            if name == n:
                return float(arr[index_map[n]])
        raise KeyError(f"Parameter name '{name}' not found among parameter names: {names}")

    # Helper to find parameter names by substring matching
    def find_param_candidates(substr, exclude_substr=None):
        cand = [n for n in names if (substr in n) and (exclude_substr is None or exclude_substr not in n)]
        return cand

    # Find main ReaderView term
    reader_matches = find_param_candidates("ReaderView", exclude_substr="Dyslexia")
    # If none, accept any name containing ReaderView (including factor-coded forms)
    if not reader_matches:
        reader_matches = [n for n in names if "ReaderView" in n]
    if not reader_matches:
        raise KeyError("Could not find a main effect parameter name for 'ReaderView' in model params: " + ", ".join(names))
    reader_name = reader_matches[0]

    # Find main Dyslexia term
    dys_matches = find_param_candidates("Dyslexia", exclude_substr="ReaderView")
    if not dys_matches:
        dys_matches = [n for n in names if "Dyslexia" in n]
    if not dys_matches:
        raise KeyError("Could not find a main effect parameter name for 'Dyslexia' in model params: " + ", ".join(names))
    dys_name = dys_matches[0]

    # Find interaction term containing both ReaderView and Dyslexia
    interaction_matches = [n for n in names if ("ReaderView" in n) and ("Dyslexia" in n)]
    # Also accept colon-style 'ReaderView:Dyslexia'
    if not interaction_matches:
        colon_name = "ReaderView:Dyslexia"
        if colon_name in names:
            interaction_matches = [colon_name]
    if not interaction_matches:
        # try the other order 'Dyslexia:ReaderView'
        colon_name_rev = "Dyslexia:ReaderView"
        if colon_name_rev in names:
            interaction_matches = [colon_name_rev]
    if not interaction_matches:
        raise KeyError("Could not find an interaction parameter name for 'ReaderView:Dyslexia' in model params: " + ", ".join(names))
    inter_name = interaction_matches[0]

    # Extract main effect (non-dyslexic)
    coef_non = get_val(params_arr, reader_name)
    se_non = get_val(bse_arr, reader_name) if bse_raw is not None else float("nan")
    t_non = coef_non / se_non if se_non != 0 and not np.isnan(se_non) else float("nan")
    p_non = get_val(pvalues_arr, reader_name) if pvalues_raw is not None else float("nan")

    # CI critical value
    if df_resid is not None and not np.isnan(df_resid):
        try:
            tcrit = stats.t.ppf(1 - 0.025, df_resid)
        except Exception:
            tcrit = stats.norm.ppf(1 - 0.025)
    else:
        tcrit = stats.norm.ppf(1 - 0.025)
    ci_non = (coef_non - tcrit * se_non, coef_non + tcrit * se_non) if not np.isnan(se_non) else (float("nan"), float("nan"))

    # Interaction coefficient
    coef_inter = get_val(params_arr, inter_name)
    se_inter = get_val(bse_arr, inter_name) if bse_raw is not None else float("nan")
    t_inter = coef_inter / se_inter if se_inter != 0 and not np.isnan(se_inter) else float("nan")
    p_inter = get_val(pvalues_arr, inter_name) if pvalues_raw is not None else float("nan")
    ci_inter = (coef_inter - tcrit * se_inter, coef_inter + tcrit * se_inter) if not np.isnan(se_inter) else (float("nan"), float("nan"))

    # Prepare covariance matrix access
    cov_mat = None
    if isinstance(cov_raw, pd.DataFrame):
        try:
            cov_mat = cov_raw.loc[[reader_name, inter_name], [reader_name, inter_name]].values
        except Exception:
            # maybe names differ; try by indices
            cov_mat = cov_raw.values
    elif isinstance(cov_raw, np.ndarray):
        cov_mat = cov_raw
    else:
        # Try to convert to array
        try:
            cov_mat = np.asarray(cov_raw)
        except Exception:
            raise RuntimeError("Could not interpret covariance matrix structure returned by cov_params().")

    # If cov_mat is full matrix ndarray and we have names -> extract submatrix by indices
    if isinstance(cov_mat, np.ndarray):
        # If cov_mat is square and size equals number of params, index by parameter indices
        if cov_mat.shape[0] == cov_mat.shape[1] == len(params_arr):
            i = index_map.get(reader_name)
            j = index_map.get(inter_name)
            if i is None or j is None:
                raise KeyError(f"Could not map parameter names to covariance matrix indices: {reader_name}, {inter_name}")
            cov_sub = cov_mat[[i, j]][:, [i, j]]
        elif cov_mat.shape == (2, 2):
            cov_sub = cov_mat
        else:
            # try to handle DataFrame-like array where first dimension includes labels - give informative error
            raise RuntimeError("Covariance matrix shape is unexpected and cannot be indexed: shape=" + str(cov_mat.shape))
    else:
        raise RuntimeError("Covariance matrix could not be converted to numpy array for indexing.")

    # Effect for dyslexic = ReaderView + ReaderView:Dyslexia
    coef_dys = coef_non + coef_inter
    var_dys = float(cov_sub[0, 0] + cov_sub[1, 1] + 2 * cov_sub[0, 1])
    se_dys = float(np.sqrt(var_dys)) if var_dys >= 0 else float("nan")
    t_dys = coef_dys / se_dys if se_dys != 0 and not np.isnan(se_dys) else float("nan")
    if df_resid is not None and not np.isnan(df_resid):
        p_dys = float(2 * stats.t.sf(abs(t_dys), df_resid)) if not np.isnan(t_dys) else float("nan")
    else:
        p_dys = float(2 * stats.norm.sf(abs(t_dys))) if not np.isnan(t_dys) else float("nan")
    ci_dys = (coef_dys - tcrit * se_dys, coef_dys + tcrit * se_dys) if not np.isnan(se_dys) else (float("nan"), float("nan"))

    # Interpret on percent-change scale (approx)
    try:
        pct_change_dys = (np.exp(coef_dys) - 1) * 100.0
    except Exception:
        pct_change_dys = float("nan")
    try:
        pct_change_non = (np.exp(coef_non) - 1) * 100.0
    except Exception:
        pct_change_non = float("nan")

    result_object = {
        "coef_non_dyslexic": coef_non,
        "se_non_dyslexic": se_non,
        "t_non_dyslexic": t_non,
        "p_non_dyslexic": p_non,
        "ci_non_dyslexic": ci_non,
        "pct_change_non_dyslexic": pct_change_non,
        "interaction_coef": coef_inter,
        "interaction_se": se_inter,
        "interaction_t": t_inter,
        "interaction_p": p_inter,
        "ci_interaction": ci_inter,
        "coef_dyslexic": coef_dys,
        "se_dyslexic": se_dys,
        "t_dyslexic": t_dys,
        "p_dyslexic": p_dys,
        "ci_dyslexic": ci_dys,
        "pct_change_dyslexic": pct_change_dys
    }

    # Human-readable description
    description = (
        "Estimated effects of activating Reader View on log(reading speed):\n"
        f"- For participants WITHOUT dyslexia (Dyslexia=0): coefficient = {coef_non:.4f} "
        f"(SE = {se_non:.4f}, t = {t_non:.2f}, p = {p_non:.3g}), 95% CI = [{ci_non[0]:.4f}, {ci_non[1]:.4f}]. "
        f"This corresponds to an approximate {pct_change_non:.2f}% change in reading speed.\n"
        f"- Interaction term (ReaderView:Dyslexia): coefficient = {coef_inter:.4f} "
        f"(SE = {se_inter:.4f}, t = {t_inter:.2f}, p = {p_inter:.3g}), 95% CI = [{ci_inter[0]:.4f}, {ci_inter[1]:.4f}].\n"
        f"- For participants WITH dyslexia (Dyslexia=1), the combined effect = ReaderView + interaction = {coef_dys:.4f} "
        f"(SE = {se_dys:.4f}, t = {t_dys:.2f}, p = {p_dys:.3g}), 95% CI = [{ci_dys[0]:.4f}, {ci_dys[1]:.4f}]. "
        f"This corresponds to an approximate {pct_change_dys:.2f}% change in reading speed.\n\n"
        "Interpretation guidance: because the outcome is log(reading speed), the coefficients can be "
        "interpreted approximately as proportional changes (exp(coef)-1). To answer whether Reader View "
        "improves reading speed for individuals with dyslexia, inspect the combined coefficient for dyslexic "
        f"participants (value above) and its p-value: a positive coefficient with a small p-value (commonly < .05) "
        "would support that Reader View increases reading speed for dyslexic participants. "
        "The interaction p-value indicates whether the ReaderView effect differs between dyslexic and non-dyslexic readers."
    )

    return {"object": result_object, "description": description}