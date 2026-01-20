def extract_final_answer(model_output):
    """
    Extract the effect of 'SkinToneBin' from a fitted (possibly clustered-wrapper) model result.

    Returns a dictionary with keys:
      - "object": dict with extracted statistics (coef, se, z, p, 95% CI, IRR, IRR 95% CI, se_type)
      - "description": short interpretation in context (whether dark-skinned players receive more red cards,
                       and whether the effect is statistically significant at alpha=0.05).

    Works with:
      - statsmodels results objects (including result of get_robustcov_results)
      - the custom ClusteredResults wrapper defined in the modeling code above
    """
    import math
    import numpy as np
    import pandas as pd

    # Helper: standard normal CDF using erf
    def std_norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # 1) Get parameter vector
    try:
        params = getattr(model_output, "params")
    except Exception:
        # try calling method
        try:
            params = model_output.params()
        except Exception:
            raise ValueError("Could not access params from model_output.")

    # Ensure params is a pandas Series (or convert)
    if isinstance(params, (np.ndarray, list)):
        # Try to get index from model_output.model.exog_names if available
        exog_names = None
        try:
            exog_names = model_output.model.exog_names
        except Exception:
            try:
                exog_names = getattr(model_output, "model").exog_names
            except Exception:
                exog_names = None
        if exog_names is not None and len(exog_names) == len(params):
            params = pd.Series(np.asarray(params).astype(float), index=list(exog_names))
        else:
            # fallback to unnamed array
            params = pd.Series(np.asarray(params).astype(float),
                               index=[f"param_{i}" for i in range(len(params))])

    if not isinstance(params, pd.Series):
        # try to coerce
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Unable to coerce params into a pandas Series.")

    # 2) Find the parameter name for SkinToneBin
    # Prefer exact name, fallback to substring (case-insensitive)
    target_name = None
    if "SkinToneBin" in params.index:
        target_name = "SkinToneBin"
    else:
        # case-insensitive search for any param containing 'skintonebin'
        matches = [n for n in params.index if "skintonebin" in str(n).lower()]
        if len(matches) == 1:
            target_name = matches[0]
        elif len(matches) > 1:
            # choose the exact match if present, else first match
            for m in matches:
                if str(m) == "SkinToneBin":
                    target_name = m
                    break
            if target_name is None:
                target_name = matches[0]

    if target_name is None:
        raise KeyError("Could not find a parameter corresponding to 'SkinToneBin' in model params. "
                       f"Available params: {list(params.index)}")

    coef = float(params[target_name])

    # 3) Obtain (clustered) covariance matrix if available, else fall back to bse
    se = None
    se_type = "unknown"
    cov_matrix = None
    try:
        # cov_params can be a method or attribute
        cov_call = getattr(model_output, "cov_params", None)
        if callable(cov_call):
            cov_matrix = cov_call()
        else:
            cov_matrix = cov_call  # maybe already a matrix
    except Exception:
        cov_matrix = None

    # If cov_matrix is a DataFrame, convert to numpy array and align index
    cov_from_df_index = None
    if isinstance(cov_matrix, pd.DataFrame):
        cov_from_df_index = list(cov_matrix.index)
        cov_matrix = cov_matrix.values

    if cov_matrix is not None and hasattr(cov_matrix, "__array__"):
        cov = np.asarray(cov_matrix, dtype=float)
        # Determine index of target parameter in covariance matrix
        try:
            if cov_from_df_index is not None:
                idx = cov_from_df_index.index(target_name)
            else:
                # try to get param order from params.index
                idx = list(params.index).index(target_name)
            var = float(cov[idx, idx])
            if var >= 0:
                se = float(math.sqrt(var))
                se_type = "clustered_cov" if cov_from_df_index is not None or se_type == "unknown" else "cov_matrix"
        except Exception:
            # fallback
            cov = None
            se = None

    # If no covariance matrix available or failed to extract, try model_output.bse
    if se is None:
        try:
            bse = getattr(model_output, "bse")
            # bse may be Series or array
            if isinstance(bse, pd.Series):
                if target_name in bse.index:
                    se = float(bse[target_name])
                    se_type = "bse_series_fallback"
                else:
                    # try substring match
                    matches = [n for n in bse.index if "skintonebin" in str(n).lower()]
                    if matches:
                        se = float(bse[matches[0]])
                        se_type = "bse_series_submatch_fallback"
            elif isinstance(bse, (np.ndarray, list)):
                # align by params order if possible
                bse_arr = np.asarray(bse, dtype=float)
                try:
                    idx = list(params.index).index(target_name)
                    se = float(bse_arr[idx])
                    se_type = "bse_array_fallback"
                except Exception:
                    # take first as last resort
                    se = float(bse_arr[0])
                    se_type = "bse_array_first_fallback"
            else:
                se = float(bse)
                se_type = "bse_scalar_fallback"
        except Exception:
            se = None

    if se is None or not np.isfinite(se) or se <= 0:
        raise RuntimeError("Could not determine a valid standard error for parameter '{}'. "
                           "Attempted clustered cov_params and bse fallbacks.".format(target_name))

    # 4) Compute z (Wald) statistic and p-value (two-sided), 95% CI, and IRR (exp(coef))
    z = coef / se
    p_value = 2.0 * (1.0 - std_norm_cdf(abs(z)))
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    irr = math.exp(coef)
    irr_ci_lower = math.exp(ci_lower)
    irr_ci_upper = math.exp(ci_upper)

    significant = bool(p_value < 0.05)

    # 5) Build output object and description
    result_obj = {
        "param_name": target_name,
        "coef": coef,
        "se": se,
        "se_type": se_type,
        "z": z,
        "p_value": p_value,
        "95%_CI_coef": [ci_lower, ci_upper],
        "IRR": irr,
        "95%_CI_IRR": [irr_ci_lower, irr_ci_upper],
        "significant_at_0.05": significant
    }

    # Interpretation: for count model with log link, coef is log rate ratio; IRR >1 means higher rate.
    if significant:
        if irr > 1.0:
            interp = (f"The estimate for '{target_name}' is positive and statistically significant (p = {p_value:.3g}), "
                      f"with an incidence rate ratio (IRR) = {irr:.3g} (95% CI: {irr_ci_lower:.3g} to {irr_ci_upper:.3g}). "
                      "This indicates that players classified as dark-skinned receive red cards at a higher rate than light-skinned players, "
                      "controlling for the listed covariates (offset = log(games)).")
        else:
            interp = (f"The estimate for '{target_name}' is negative and statistically significant (p = {p_value:.3g}), "
                      f"with IRR = {irr:.3g} (95% CI: {irr_ci_lower:.3g} to {irr_ci_upper:.3g}). "
                      "This indicates dark-skinned players receive red cards at a lower rate than light-skinned players, "
                      "controlling for the listed covariates.")
    else:
        # Not significant
        interp = (f"The estimate for '{target_name}' is not statistically significant at alpha=0.05 (p = {p_value:.3g}). "
                  f"Point estimate IRR = {irr:.3g} (95% CI: {irr_ci_lower:.3g} to {irr_ci_upper:.3g}). "
                  "This means we do not have strong evidence that dark-skinned players differ from light-skinned players in red card rates, "
                  "after adjustment for covariates and with the reported standard-error method ({se_type}).").format(se_type=se_type)

    return {"object": result_obj, "description": interp}