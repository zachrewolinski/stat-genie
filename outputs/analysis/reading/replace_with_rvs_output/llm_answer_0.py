def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View for dyslexic readers from the provided
    model_output and returns a concise numeric summary and interpretation.

    Returns a dict with keys:
      - "object": dict with numeric results (effect, SE, t, p, 95% CI)
      - "description": short plain-language interpretation answering whether Reader View
                       improves reading speed for dyslexic individuals.
    """
    import numpy as np
    from scipy import stats

    # Helper to build the results dict
    def build_result(effect, se, t_stat, p_val):
        if se is not None and not np.isnan(se):
            ci_low = effect - 1.96 * se
            ci_high = effect + 1.96 * se
        else:
            ci_low = ci_high = None
        return {
            "effect_reader_view_for_dyslexic": effect,
            "se_effect_reader_view_for_dyslexic": se,
            "t_effect_reader_view_for_dyslexic": t_stat,
            "p_effect_reader_view_for_dyslexic": p_val,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high
        }

    # Try to extract pre-computed summary from a dict-like model_output
    effect = se = t_stat = p_val = None
    try:
        if isinstance(model_output, dict):
            # If the caller already returned the summary fields
            if "effect_reader_view_for_dyslexic" in model_output:
                effect = float(model_output.get("effect_reader_view_for_dyslexic"))
                se = model_output.get("se_effect_reader_view_for_dyslexic")
                se = float(se) if se is not None else None
                t_stat = model_output.get("t_effect_reader_view_for_dyslexic")
                t_stat = float(t_stat) if t_stat is not None else None
                p_val = model_output.get("p_effect_reader_view_for_dyslexic")
                p_val = float(p_val) if p_val is not None else None
            # Otherwise, try to extract coefficients from a wrapped model result
            elif "model_result" in model_output and hasattr(model_output["model_result"], "params"):
                res = model_output["model_result"]
                params = getattr(res, "params")
                cov = None
                try:
                    cov = res.cov_params()
                except Exception:
                    cov = None
                b_rv = float(params.get("reader_view", 0.0))
                b_int = float(params.get("reader_view_x_dyslexia", 0.0))
                effect = b_rv + b_int
                # compute SE if covariance available
                if cov is not None:
                    try:
                        v_rv = float(cov.loc["reader_view", "reader_view"])
                        v_int = float(cov.loc["reader_view_x_dyslexia", "reader_view_x_dyslexia"])
                        covar = float(cov.loc["reader_view", "reader_view_x_dyslexia"])
                        se = float(np.sqrt(v_rv + v_int + 2 * covar))
                    except Exception:
                        se = None
                else:
                    se = None
                if se is not None and se > 0:
                    t_stat = effect / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                else:
                    t_stat = None
                    p_val = None
        # If model_output itself is a statsmodels results object
        elif hasattr(model_output, "params"):
            res = model_output
            params = getattr(res, "params")
            cov = None
            try:
                cov = res.cov_params()
            except Exception:
                cov = None
            b_rv = float(params.get("reader_view", 0.0))
            b_int = float(params.get("reader_view_x_dyslexia", 0.0))
            effect = b_rv + b_int
            if cov is not None:
                try:
                    v_rv = float(cov.loc["reader_view", "reader_view"])
                    v_int = float(cov.loc["reader_view_x_dyslexia", "reader_view_x_dyslexia"])
                    covar = float(cov.loc["reader_view", "reader_view_x_dyslexia"])
                    se = float(np.sqrt(v_rv + v_int + 2 * covar))
                except Exception:
                    se = None
            else:
                se = None
            if se is not None and se > 0:
                t_stat = effect / se
                p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            else:
                t_stat = None
                p_val = None
    except Exception:
        # If anything goes wrong, return an informative message below
        effect = se = t_stat = p_val = None

    # If we still don't have numeric values, try to find individual coefficients in a dict
    if effect is None and isinstance(model_output, dict):
        try:
            b_rv = model_output.get("coef_reader_view")
            b_int = model_output.get("coef_interaction_reader_view_x_dyslexia")
            if b_rv is not None and b_int is not None:
                effect = float(b_rv) + float(b_int)
                se = model_output.get("se_effect_reader_view_for_dyslexic")
                se = float(se) if se is not None else None
                t_stat = model_output.get("t_effect_reader_view_for_dyslexic")
                t_stat = float(t_stat) if t_stat is not None else None
                p_val = model_output.get("p_effect_reader_view_for_dyslexic")
                p_val = float(p_val) if p_val is not None else None
        except Exception:
            pass

    results = build_result(effect, se, t_stat, p_val)

    # Build concise interpretation
    if effect is None:
        description = ("Could not extract the estimated effect of Reader View for dyslexic readers "
                       "from the provided model_output. Please provide either a dict containing "
                       "'effect_reader_view_for_dyslexic' (and optionally SE/t/p) or a fitted "
                       "statsmodels results object with parameters named 'reader_view' and "
                       "'reader_view_x_dyslexia'.")
    else:
        # Interpret significance
        sig = None
        if p_val is None:
            sig = "inconclusive (p-value not available)"
        else:
            if p_val < 0.05 and effect > 0:
                sig = f"statistically significant improvement (p = {p_val:.3g})"
            elif p_val < 0.05 and effect <= 0:
                sig = f"statistically significant change but not an improvement (effect <= 0; p = {p_val:.3g})"
            else:
                sig = f"no evidence of a statistically significant effect (p = {p_val:.3g})"

        # Make the text concise and focused on yes/no
        if p_val is not None and p_val < 0.05 and effect > 0:
            verdict = "Yes — there is evidence that Reader View improves reading speed for dyslexic readers."
        elif p_val is not None and p_val >= 0.05:
            verdict = "No — there is no evidence that Reader View improves reading speed for dyslexic readers."
        else:
            verdict = "Inconclusive — could not determine whether Reader View improves reading speed for dyslexic readers."

        # Compose description including numeric summary
        ci_text = ""
        if results["ci_95_lower"] is not None:
            ci_text = f" 95% CI [{results['ci_95_lower']:.2f}, {results['ci_95_upper']:.2f}]."
        description = (
            f"Estimated effect of Reader View for dyslexic readers = {results['effect_reader_view_for_dyslexic']:.4f} "
            f"(SE = {results['se_effect_reader_view_for_dyslexic']:.4f}, t = "
            f"{results['t_effect_reader_view_for_dyslexic']:.4f}, p = {results['p_effect_reader_view_for_dyslexic']:.4f})."
            + ci_text + " " + verdict + f" ({sig})"
        )

    return {"object": results, "description": description}