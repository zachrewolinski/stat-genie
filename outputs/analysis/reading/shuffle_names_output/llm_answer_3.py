def extract_final_answer(model_output):
    """
    Extract statistics relevant to the question:
      "Does Reader View improve reading speed for individuals with dyslexia?"

    Expects a statsmodels RegressionResults-like object (with .params, .bse, .pvalues,
    .conf_int(), and .cov_params() ideally). Handles None and tries to be robust
    to slightly different term naming (e.g., 'reader_view_on:is_dyslexic').

    Returns a dictionary with keys:
      - "object": dict of numeric results (coefficients, p-values, CIs) for:
          * reader_view effect for non-dyslexic (main effect)
          * reader_view effect for dyslexic (main + interaction)
          * raw coefficients for main and interaction terms
          * a booleanish conclusion and p-values
      - "description": short human-readable interpretation.
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy.stats import norm
    except Exception:
        # fallback: implement normal cdf/ppf via numpy if scipy not available (approx)
        def _erf(x):
            # approximation of error function
            # Abramowitz and Stegun formula 7.1.26
            sign = np.sign(x)
            x = np.abs(x)
            t = 1.0 / (1.0 + 0.3275911 * x)
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
            return sign * y
        def _norm_cdf(x):
            return 0.5 * (1 + _erf(x / np.sqrt(2)))
        class _norm:
            @staticmethod
            def sf(x):
                return 1.0 - _norm_cdf(x)
            @staticmethod
            def ppf(q):
                # rough inverse via numpy (not very accurate). Try using scipy if possible.
                # fallback: use approximation for median and scale
                from math import sqrt
                return np.sqrt(2) * np.erfinv(2*q - 1) if hasattr(np, 'erfinv') else 0.0
        norm = _norm

    # Helper to build a standard small result dictionary in case of problems
    def _empty_result(msg):
        return {
            "object": None,
            "description": f"Could not extract results: {msg}"
        }

    if model_output is None:
        return _empty_result("model_output is None")

    # Try to detect statsmodels-like results
    try:
        params = getattr(model_output, "params", None)
        if params is None:
            # maybe the model_output is a fitted sklearn-like object: no extraction supported
            return _empty_result("model_output has no .params attribute (not a statsmodels result).")
        # Make params a pandas Series or dict-like
        # Convert index to strings for searching
        param_index = list(params.index) if hasattr(params, 'index') else list(params.keys())
        # Find candidate names for main and interaction terms
        main_name = None
        interaction_name = None
        dys_name = None
        # Prefer exact simple names if present
        for name in param_index:
            n = str(name)
            if n == 'reader_view_on':
                main_name = n
            if n == 'is_dyslexic':
                dys_name = n
            if (('reader_view_on' in n) and ('is_dyslexic' in n) and (':' in n or '*' in n or 'reader_view_on' in n and 'is_dyslexic' in n and n.count(':')==1)):
                interaction_name = n
        # If exact not found, use heuristics
        if main_name is None:
            # pick a param that contains reader_view_on but not ':' (main effect)
            for name in param_index:
                n = str(name)
                if ('reader_view_on' in n) and (':' not in n):
                    main_name = n
                    break
        if interaction_name is None:
            # pick a param that contains both names
            for name in param_index:
                n = str(name)
                if ('reader_view_on' in n) and ('is_dyslexic' in n) and (':' in n or '.' in n or '_' in n):
                    interaction_name = n
                    break
        if dys_name is None:
            for name in param_index:
                n = str(name)
                if ('is_dyslexic' in n) and (':' not in n):
                    dys_name = n
                    break

        # If we still can't find a main reader_view_on term, abort
        if main_name is None:
            return _empty_result("Could not locate a 'reader_view_on' term in model parameters.")

        # Extract values
        coef_main = float(params[main_name])
        # p-value for main if available
        pvals = getattr(model_output, "pvalues", None)
        p_main = float(pvals[main_name]) if (pvals is not None and main_name in pvals.index) else None

        # Confidence intervals if available
        try:
            ci_df = model_output.conf_int()
            if main_name in ci_df.index:
                ci_main = (float(ci_df.loc[main_name, 0]), float(ci_df.loc[main_name, 1]))
            else:
                ci_main = None
        except Exception:
            ci_main = None

        # Interaction term details (may be absent -> effect for dyslexic equals main)
        if interaction_name is not None and interaction_name in params.index:
            coef_int = float(params[interaction_name])
            p_int = float(pvals[interaction_name]) if (pvals is not None and interaction_name in pvals.index) else None
            try:
                ci_int = (float(ci_df.loc[interaction_name, 0]), float(ci_df.loc[interaction_name, 1])) if ci_df is not None and interaction_name in ci_df.index else None
            except Exception:
                ci_int = None
        else:
            coef_int = 0.0
            p_int = None
            ci_int = None

        # Compute effect of reader_view_on for dyslexic = main + interaction
        coef_dys = coef_main + coef_int

        # To get SE for the sum, use covariance matrix if available
        se_dys = None
        try:
            cov = model_output.cov_params()
            # Ensure both terms present in cov
            if (main_name in cov.index) and (interaction_name in cov.index):
                var_sum = cov.loc[main_name, main_name] + cov.loc[interaction_name, interaction_name] + 2 * cov.loc[main_name, interaction_name]
                se_dys = float(np.sqrt(max(var_sum, 0.0)))
            else:
                # If interaction_name missing from cov (e.g., no interaction term), fall back to bse
                bse = getattr(model_output, "bse", None)
                if bse is not None and main_name in bse.index:
                    # if no interaction, se is main bse
                    se_dys = float(bse[main_name])
                else:
                    se_dys = None
        except Exception:
            # fallback: use bse values and ignore covariance
            bse = getattr(model_output, "bse", None)
            if bse is not None:
                try:
                    b_main = float(bse[main_name]) if main_name in bse.index else None
                except Exception:
                    b_main = None
                try:
                    b_int = float(bse[interaction_name]) if (bse is not None and interaction_name in bse.index) else 0.0
                except Exception:
                    b_int = 0.0
                if b_main is not None:
                    se_dys = float(sqrt(max(b_main**2 + b_int**2, 0.0)))
                else:
                    se_dys = None
            else:
                se_dys = None

        # p-value and CI for dyslexic effect using normal approximation
        if se_dys is not None and se_dys > 0:
            z_dys = coef_dys / se_dys
            p_dys = float(2.0 * norm.sf(abs(z_dys)))
            zcrit = norm.ppf(0.975)
            ci_dys = (float(coef_dys - zcrit * se_dys), float(coef_dys + zcrit * se_dys))
        else:
            p_dys = None
            ci_dys = None

        # For non-dyslexic (is_dyslexic=0), effect is just coef_main (we already have p_main and ci_main)
        # If p_main missing but bse available, compute p_main similarly
        if p_main is None:
            try:
                bse = getattr(model_output, "bse", None)
                if bse is not None and main_name in bse.index and bse[main_name] > 0:
                    z_main = coef_main / float(bse[main_name])
                    p_main = float(2.0 * norm.sf(abs(z_main)))
                else:
                    p_main = None
            except Exception:
                p_main = None

        if ci_main is None:
            try:
                bse = getattr(model_output, "bse", None)
                if bse is not None and main_name in bse.index:
                    zcrit = norm.ppf(0.975)
                    se_main = float(bse[main_name])
                    ci_main = (float(coef_main - zcrit * se_main), float(coef_main + zcrit * se_main))
                else:
                    ci_main = None
            except Exception:
                ci_main = None

        # Determine conclusion: focus on dyslexic effect (coef_dys > 0 and p < 0.05)
        conclusion = {
            "improves_for_dyslexic": None,
            "conclusion_text": None
        }
        if p_dys is not None:
            improves = (coef_dys > 0) and (p_dys < 0.05)
            conclusion["improves_for_dyslexic"] = bool(improves)
            if improves:
                conclusion["conclusion_text"] = (
                    f"Reader View appears to significantly increase reading speed for dyslexic participants "
                    f"(estimated effect = {coef_dys:.3f} WPM, p = {p_dys:.3g})."
                )
            else:
                conclusion["conclusion_text"] = (
                    f"No statistically significant evidence that Reader View improves reading speed for dyslexic participants "
                    f"(estimated effect = {coef_dys:.3f} WPM, p = {p_dys:.3g})."
                )
        else:
            conclusion["improves_for_dyslexic"] = None
            conclusion["conclusion_text"] = "Could not compute a p-value for the dyslexic subgroup effect."

        result_obj = {
            "coef_main_reader_view_non_dyslexic": coef_main,
            "p_main_reader_view_non_dyslexic": p_main,
            "ci_main_reader_view_non_dyslexic": ci_main,
            "coef_interaction_reader_view_x_dyslexic": coef_int,
            "p_interaction": p_int,
            "ci_interaction": ci_int,
            "coef_reader_view_for_dyslexic": coef_dys,
            "p_reader_view_for_dyslexic": p_dys,
            "ci_reader_view_for_dyslexic": ci_dys,
            "conclusion": conclusion
        }

        # Build a short human-readable description
        if conclusion["improves_for_dyslexic"] is True:
            desc = (
                f"Estimated Reader View effect for dyslexic participants: {coef_dys:.3f} WPM "
                f"(95% CI [{ci_dys[0]:.3f}, {ci_dys[1]:.3f}]), p = {p_dys:.3g}. "
                "This indicates a statistically significant improvement in reading speed for dyslexic readers."
            )
        elif conclusion["improves_for_dyslexic"] is False:
            desc = (
                f"Estimated Reader View effect for dyslexic participants: {coef_dys:.3f} WPM "
                f"(95% CI [{(ci_dys[0] if ci_dys else float('nan')):.3f}, {(ci_dys[1] if ci_dys else float('nan')):.3f}]), p = {p_dys:.3g}. "
                "This provides no evidence of a statistically significant improvement for dyslexic readers."
            )
        else:
            desc = conclusion["conclusion_text"]

        return {
            "object": result_obj,
            "description": desc
        }

    except Exception as e:
        return _empty_result(f"Exception while extracting results: {e}")