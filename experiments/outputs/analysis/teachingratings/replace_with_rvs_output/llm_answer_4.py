def extract_final_answer(model_output):
    """
    Extract statistics relating to the effect of instructor beauty on student evaluations
    from a fitted statsmodels OLS results object (with robust/clustered cov).

    Returns a dictionary with keys:
      - "object": a dict containing coefficients, SEs, p-values, 95% CIs for:
            * linear beauty term (beauty_c)
            * quadratic beauty term (beauty_sq)
            * interaction of beauty with gender (beauty_c:gender_female) if present
        plus marginal (derivative) effects of beauty at:
            * mean (beauty_c = 0) for male and female instructors
            * ±1 SD of beauty_c for male and female instructors
        Each reported effect includes estimate, std err, t-stat, p-value, and 95% CI (where computable).
      - "description": a human-readable explanation of what the returned numbers mean.
    """
    import numpy as np
    import pandas as pd

    res = model_output  # statsmodels result object (robust results wrapper)

    # Raw outputs (may be numpy arrays or pandas objects)
    raw_params = getattr(res, "params", None)
    raw_pvalues = getattr(res, "pvalues", None)
    raw_bse = getattr(res, "bse", None)
    raw_conf = None
    try:
        raw_conf = res.conf_int()
    except Exception:
        raw_conf = None

    # Determine parameter names robustly
    if raw_params is None:
        raise ValueError("Model result object has no 'params' attribute.")

    if hasattr(raw_params, "index"):
        param_names = list(raw_params.index)
    else:
        # Try to get names from the model object
        param_names = None
        if hasattr(res, "model"):
            if hasattr(res.model, "exog_names"):
                try:
                    param_names = list(res.model.exog_names)
                except Exception:
                    param_names = None
            # some versions store param_names in data
            if param_names is None and hasattr(res.model, "data"):
                data = res.model.data
                if hasattr(data, "param_names"):
                    try:
                        param_names = list(data.param_names)
                    except Exception:
                        param_names = None
        # Fallback to generic names
        if param_names is None:
            try:
                length = int(np.asarray(raw_params).ravel().shape[0])
            except Exception:
                length = 0
            param_names = [f"param_{i}" for i in range(length)]

    # Convert to pandas Series/DataFrame for easier indexing
    params = pd.Series(np.asarray(raw_params).ravel(), index=param_names)

    if raw_pvalues is None:
        pvalues = pd.Series([np.nan] * len(param_names), index=param_names)
    elif hasattr(raw_pvalues, "index"):
        pvalues = pd.Series(np.asarray(raw_pvalues).ravel(), index=list(raw_pvalues.index))
    else:
        pvalues = pd.Series(np.asarray(raw_pvalues).ravel(), index=param_names)

    if raw_bse is None:
        bse = pd.Series([np.nan] * len(param_names), index=param_names)
    elif hasattr(raw_bse, "index"):
        bse = pd.Series(np.asarray(raw_bse).ravel(), index=list(raw_bse.index))
    else:
        bse = pd.Series(np.asarray(raw_bse).ravel(), index=param_names)

    if raw_conf is None:
        conf_int_df = None
    elif hasattr(raw_conf, "index"):
        try:
            conf_int_df = pd.DataFrame(raw_conf)
        except Exception:
            conf_int_df = None
    else:
        # raw_conf might be an ndarray with shape (k,2)
        try:
            arr = np.asarray(raw_conf)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == len(param_names):
                conf_int_df = pd.DataFrame(arr, index=param_names)
            else:
                conf_int_df = None
        except Exception:
            conf_int_df = None

    # Helper to find parameter name among possible alternatives
    def find_name(targets):
        for t in targets:
            if t in params.index:
                return t
        return None

    beauty_name = find_name(["beauty_c"])
    beauty_sq_name = find_name(["beauty_sq"])
    gender_name = find_name(["gender_female"])
    interaction_name = find_name(["beauty_c:gender_female", "gender_female:beauty_c"])

    if beauty_name is None or beauty_sq_name is None:
        raise KeyError("Required beauty terms not found in model parameters.")

    # Helper to format single-parameter results
    def single_param_stats(name):
        est = float(params.get(name, np.nan))
        se = float(bse.get(name, np.nan)) if name in bse.index else np.nan
        tstat = est / se if (se != 0 and not np.isnan(se)) else np.nan
        pval = float(pvalues.get(name, np.nan)) if name in pvalues.index else np.nan
        if conf_int_df is not None and name in conf_int_df.index:
            try:
                ci_lower, ci_upper = float(conf_int_df.loc[name, 0]), float(conf_int_df.loc[name, 1])
            except Exception:
                ci_lower, ci_upper = np.nan, np.nan
        else:
            ci_lower, ci_upper = np.nan, np.nan
        return {"estimate": est, "std_err": se, "t": tstat, "p_value": pval, "ci_95": (ci_lower, ci_upper)}

    # Collect main parameter stats
    results = {}
    results["beauty_c"] = single_param_stats(beauty_name)
    results["beauty_sq"] = single_param_stats(beauty_sq_name)
    if gender_name is not None and gender_name in params.index:
        results["gender_female"] = single_param_stats(gender_name)
    if interaction_name is not None:
        results["interaction"] = single_param_stats(interaction_name)
    else:
        # If no explicit interaction, we treat it as zero
        results["interaction"] = {
            "estimate": 0.0,
            "std_err": 0.0,
            "t": np.nan,
            "p_value": np.nan,
            "ci_95": (np.nan, np.nan),
        }

    # Function to compute linear-combination test using res.t_test (keeps correct cov matrix & df)
    def linear_combination_test(expr):
        """
        expr: a string expression using parameter names, e.g. "beauty_c + 2*beauty_sq"
        returns dict with estimate, std_err, t, p_value, ci_95
        """
        try:
            tt = res.t_test(expr)
            # Extract effect, sd, tvalue, pvalue robustly
            eff = np.asarray(getattr(tt, "effect", np.asarray([np.nan]))).ravel()
            est = float(eff[0]) if eff.size > 0 else np.nan

            sd_arr = None
            if hasattr(tt, "sd"):
                sd_arr = np.asarray(tt.sd).ravel()
            elif hasattr(tt, "sd_obs"):
                sd_arr = np.asarray(tt.sd_obs).ravel()
            se = float(sd_arr[0]) if (sd_arr is not None and sd_arr.size > 0) else np.nan

            t_arr = None
            if hasattr(tt, "tvalue"):
                t_arr = np.asarray(tt.tvalue).ravel()
            elif hasattr(tt, "tobs"):
                t_arr = np.asarray(tt.tobs).ravel()
            tval = float(t_arr[0]) if (t_arr is not None and t_arr.size > 0) else (est / se if (se != 0 and not np.isnan(se)) else np.nan)

            # pvalue may be array-like or scalar
            p_raw = getattr(tt, "pvalue", None)
            if p_raw is None:
                p_arr = None
                if hasattr(tt, "prob"):
                    p_arr = np.asarray(tt.prob).ravel()
                pval = float(p_arr[0]) if (p_arr is not None and p_arr.size > 0) else np.nan
            else:
                p_arr = np.asarray(p_raw).ravel()
                pval = float(p_arr[0]) if p_arr.size > 0 else np.nan

            # Compute 95% CI from estimate +/- t_crit * se using df_resid if possible
            try:
                df_resid = float(getattr(res, "df_resid"))
                # critical t
                from scipy.stats import t as _t

                tcrit = float(_t.ppf(0.975, df_resid))
                ci_lower = est - tcrit * se
                ci_upper = est + tcrit * se
            except Exception:
                # fallback to normal approx if scipy not available
                ci_lower = est - 1.96 * se
                ci_upper = est + 1.96 * se

            return {"estimate": est, "std_err": se, "t": tval, "p_value": pval, "ci_95": (ci_lower, ci_upper)}
        except Exception as e:
            return {"error": f"t_test failed for expression '{expr}': {e}"}

    # Marginal effect (derivative) is: d(eval)/d(beauty_c) = beta_beauty + 2 * beta_beauty_sq * beauty_c + beta_interaction * gender_female
    # We report this at beauty_c = 0 (mean, since beauty_c is centered), and at +/- 1 SD of beauty_c if data is available.
    # Build expressions for male (gender_female=0) and female (gender_female=1).
    # At mean (beauty_c = 0)
    expr_male_mean = f"{beauty_name}"
    if interaction_name is not None:
        expr_female_mean = f"{beauty_name} + {interaction_name}"
    else:
        expr_female_mean = f"{beauty_name}"

    results["marginal_at_mean"] = {
        "male": linear_combination_test(expr_male_mean),
        "female": linear_combination_test(expr_female_mean),
    }

    # Attempt to get +/-1 SD of beauty_c from the model's data frame
    sd_available = False
    sd = None
    try:
        model_df = getattr(res.model, "data", None)
        if model_df is not None and hasattr(model_df, "frame") and model_df.frame is not None:
            df = model_df.frame
            if beauty_name in df.columns:
                sd = float(df[beauty_name].std(ddof=0))
                sd_available = True
        elif model_df is not None and hasattr(model_df, "orig_endog") and hasattr(model_df, "orig_exog"):
            # last-resort: check exog dataframe columns if provided
            df = model_df.frame if hasattr(model_df, "frame") else None
            if df is not None and beauty_name in df.columns:
                sd = float(df[beauty_name].std(ddof=0))
                sd_available = True
        # else leave sd_available False
    except Exception:
        sd_available = False

    if sd_available and sd is not None and sd > 0:
        # plus 1 SD
        expr_male_plus1sd = f"{beauty_name} + 2*{beauty_sq_name}*{sd}"
        if interaction_name is not None:
            expr_female_plus1sd = f"{beauty_name} + 2*{beauty_sq_name}*{sd} + {interaction_name}"
        else:
            expr_female_plus1sd = expr_male_plus1sd

        expr_male_minus1sd = f"{beauty_name} + 2*{beauty_sq_name}*{-sd}"
        if interaction_name is not None:
            expr_female_minus1sd = f"{beauty_name} + 2*{beauty_sq_name}*{-sd} + {interaction_name}"
        else:
            expr_female_minus1sd = expr_male_minus1sd

        results["marginal_at_plus_1sd"] = {
            "sd_value": sd,
            "male": linear_combination_test(expr_male_plus1sd),
            "female": linear_combination_test(expr_female_plus1sd),
        }
        results["marginal_at_minus_1sd"] = {
            "sd_value": sd,
            "male": linear_combination_test(expr_male_minus1sd),
            "female": linear_combination_test(expr_female_minus1sd),
        }
    else:
        results["marginal_at_plus_1sd"] = "beauty_c SD not available in model data; cannot compute +/-1SD marginal effects"
        results["marginal_at_minus_1sd"] = "beauty_c SD not available in model data; cannot compute +/-1SD marginal effects"

    # Package final return object
    description_lines = [
        "Returned values include estimates, standard errors, t-statistics, p-values, and 95% CIs for:",
        "- The linear beauty term (beauty_c)",
        "- The quadratic beauty term (beauty_sq)",
        "- The interaction of beauty with female gender (if present)",
        "",
        "Also included are marginal effects (derivatives of eval w.r.t. beauty):",
        "- At the centered mean of beauty_c (0): for male (gender_female=0) and female (gender_female=1).",
        "- At +/- 1 SD of beauty_c (if the original model data frame is available to compute SD).",
        "",
        "Interpretation guidance:",
        "- If the marginal effect estimate is positive and the p-value is small (e.g., < 0.05), higher beauty is associated with higher evaluations.",
        "- If the quadratic term is significant, the marginal effect changes with beauty; use the provided marginal_at_plus/minus_1sd results to see curvature.",
        "- The female marginal effect adds the interaction coefficient to the male effect; the t_test computed for that linear combination (female) accounts for covariance between coefficients.",
    ]
    description = "\n".join(description_lines)

    return {"object": results, "description": description}