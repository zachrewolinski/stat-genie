def extract_final_answer(model_output):
    """
    Extracts the estimated Reader View effect for dyslexic readers from a fitted statsmodels OLS result.
    Returns a dict with:
      - "object": dict with numeric results (combined coefficient on log_speed, SE, t, p, 95% CI,
                  and percent change interpretation)
      - "description": plain-language explanation of the numbers and what they mean for the task.

    The function is robust to:
      - model_output being None (returns an explanatory message)
      - slight variations in parameter names (will try to locate suitable parameter names)
      - models fit with clustered cov (uses model's t_test when available)
    """
    import numpy as np

    if model_output is None:
        return {
            "object": None,
            "description": "No model object was provided (model_output is None). Cannot extract statistics."
        }

    # Helper: try to find a parameter name in the model params that matches expected patterns.
    def find_param_name(params_index, candidates):
        # params_index: iterable of parameter names
        # candidates: list of exact names to try in order; if none found, try fuzzy match
        for c in candidates:
            if c in params_index:
                return c
        # fuzzy: look for any param name that contains all tokens from candidate split by non-alphanum
        for c in candidates:
            tokens = [t for t in ''.join(ch if ch.isalnum() else ' ' for ch in c).split() if t]
            for pname in params_index:
                pname_low = pname.lower()
                if all(tok.lower() in pname_low for tok in tokens):
                    return pname
        return None

    try:
        params_index = list(model_output.params.index)
    except Exception:
        # fallback: try getattr
        try:
            params_index = list(model_output.params.keys())
        except Exception:
            return {
                "object": None,
                "description": "Could not access parameter names from model_output. Ensure this is a fitted statsmodels result."
            }

    # Expected parameter names from formula: 'reader_view' and 'reader_view:dyslexia_bin'
    main_candidates = ['reader_view']
    inter_candidates = ['reader_view:dyslexia_bin', 'reader_view:dyslexia_bin', 'reader_view:dyslexia_bin[1]', 'reader_view:dyslexia_bin[T.1]']

    main_name = find_param_name(params_index, main_candidates)
    inter_name = find_param_name(params_index, inter_candidates)

    # If main term not found, try fuzzy match for anything containing 'reader' and 'view'
    if main_name is None:
        main_name = find_param_name(params_index, ['reader view', 'readerview', 'reader_view'])

    if main_name is None:
        return {
            "object": None,
            "description": "Could not find a parameter corresponding to the Reader View main effect in the model parameters."
        }

    # Build contrast vector: effect for dyslexic readers = main + interaction (if interaction present)
    # If interaction not present, the effect for dyslexic = main (no moderator)
    try:
        k_params = len(params_index)
        contrast = np.zeros((k_params,))
        main_idx = params_index.index(main_name)
        contrast[main_idx] = 1.0
        used_terms = [main_name]
        if inter_name is not None and inter_name in params_index:
            inter_idx = params_index.index(inter_name)
            contrast[inter_idx] = 1.0
            used_terms.append(inter_name)
    except Exception as e:
        return {
            "object": None,
            "description": f"Error constructing contrast vector for linear combination: {e}"
        }

    # Use model_output.t_test if available (it uses the model covariance matrix, including clustering)
    try:
        ttest_res = model_output.t_test(contrast)
        # ttest_res.effect may be array-like; convert to scalar
        effect = float(np.squeeze(ttest_res.effect))
        se = float(np.squeeze(ttest_res.sd))
        tvalue = float(np.squeeze(ttest_res.tvalue))
        # pvalue can be array or scalar; ensure scalar
        pvalue = float(np.squeeze(ttest_res.pvalue))
        # confidence interval: t_test returns shape (1,2)
        try:
            ci = np.squeeze(ttest_res.conf_int(alpha=0.05))
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            # fallback to using params +/- 1.96*se (large-sample)
            ci_lower, ci_upper = effect - 1.96 * se, effect + 1.96 * se
    except Exception:
        # fallback: compute effect and SE manually from params and covariance matrix
        try:
            params = model_output.params.values
            cov = model_output.cov_params()
            # ensure cov is a numpy array aligned with params_index
            cov_mat = np.asarray(cov)
            effect = float(np.dot(contrast, params))
            var_lincomb = float(contrast @ cov_mat @ contrast)
            se = float(np.sqrt(var_lincomb))
            tvalue = float(effect / se) if se != 0 else float('nan')
            # two-sided p-value using large-sample normal approximation if df not available
            try:
                from scipy import stats
                if hasattr(model_output, 'df_resid') and np.isfinite(model_output.df_resid):
                    pvalue = float(2 * stats.t.sf(abs(tvalue), df=model_output.df_resid))
                else:
                    pvalue = float(2 * stats.norm.sf(abs(tvalue)))
            except Exception:
                # no scipy: use normal approx
                pvalue = float(2 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(tvalue) / np.sqrt(2.0)))))
            ci_lower, ci_upper = effect - 1.96 * se, effect + 1.96 * se
        except Exception as e:
            return {
                "object": None,
                "description": f"Failed to compute linear combination and its SE: {e}"
            }

    # Interpret effect in original speed units: outcome is log_speed, so exp(effect)-1 gives multiplicative change.
    try:
        effect_pct = (np.exp(effect) - 1.0) * 100.0
        ci_lower_pct = (np.exp(ci_lower) - 1.0) * 100.0
        ci_upper_pct = (np.exp(ci_upper) - 1.0) * 100.0
    except Exception:
        effect_pct = None
        ci_lower_pct = None
        ci_upper_pct = None

    result_object = {
        "terms_used": used_terms,
        "coef_log_speed": effect,
        "se": se,
        "t": tvalue,
        "p_value": pvalue,
        "ci_95_log_speed": [ci_lower, ci_upper],
        "percent_change_speed": effect_pct,  # percent increase in speed for dyslexic readers when Reader View ON
        "ci_95_percent_change": [ci_lower_pct, ci_upper_pct],
        "interpretation_note": (
            "The coefficient is the estimated effect of turning Reader View ON for readers with dyslexia "
            "(i.e., the sum of the Reader View main effect and the ReaderView x Dyslexia interaction). "
            "Because the dependent variable is log(speed), exp(coef)-1 gives the multiplicative change in reading speed."
        )
    }

    # Short plain-language description
    if pvalue <= 0.05:
        significance = "statistically significant (p <= 0.05)"
    elif pvalue <= 0.1:
        significance = "marginally significant (0.05 < p <= 0.1)"
    else:
        significance = "not statistically significant (p > 0.1)"

    description = (
        f"Estimated effect of Reader View for participants with dyslexia (linear combination of {', '.join(used_terms)}):\n"
        f"- Coefficient on log_speed = {effect:.4f} (SE = {se:.4f}, t = {tvalue:.3f}, p = {pvalue:.4f}). This is {significance}.\n"
        f"- 95% CI on log scale: [{ci_lower:.4f}, {ci_upper:.4f}].\n"
        f"- Interpreted on the original speed scale: estimated change = {effect_pct:.2f}% "
        f"(95% CI: [{ci_lower_pct:.2f}%, {ci_upper_pct:.2f}%]).\n\n"
        "If the percent change is positive and statistically significant, it indicates Reader View improves reading speed for individuals with dyslexia. "
        "If it's negative and significant, it indicates a decrease in reading speed. If not significant, the data do not provide strong evidence of an effect."
    )

    return {"object": result_object, "description": description}