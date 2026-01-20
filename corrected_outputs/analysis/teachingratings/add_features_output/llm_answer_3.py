def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, p-value, 95% CI, and a simple
    significance verdict for the 'beauty_z' variable from a fitted
    statsmodels regression results object (or a clustered-robust wrapper).
    
    Returns:
      dict with keys:
        - "object": dict containing numeric results:
            {
              "coef": float,
              "std_err": float,
              "p_value": float,
              "ci_lower": float,
              "ci_upper": float,
              "nobs": int or None,
              "significant_0.05": bool
            }
        - "description": str human-readable interpretation in the context:
            what the coefficient means (change in Eval per 1 SD in beauty),
            its statistical significance, and the 95% CI.
    """
    # Defensive imports (no external required, but keep reference)
    import math

    # Prepare a helper to safely get series-like entries
    def _get(series_like, key):
        try:
            # If it's a pandas Series or dict-like
            return series_like[key]
        except Exception:
            try:
                # If key isn't present as string but as index position (unlikely)
                return series_like.loc[key]
            except Exception:
                return None

    out = {
        "coef": None,
        "std_err": None,
        "p_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "nobs": None,
        "significant_0.05": None
    }

    try:
        res = model_output  # alias

        # params (coefficients)
        params = getattr(res, "params", None)
        bse = getattr(res, "bse", None)
        pvalues = getattr(res, "pvalues", None)
        conf_int = None
        try:
            conf_int = res.conf_int()
        except Exception:
            conf_int = None

        # Attempt to extract values for 'beauty_z'
        coef = _get(params, 'beauty_z') if params is not None else None
        std_err = _get(bse, 'beauty_z') if bse is not None else None
        pval = _get(pvalues, 'beauty_z') if pvalues is not None else None
        ci_lower = ci_upper = None
        if conf_int is not None:
            try:
                row = conf_int.loc['beauty_z']
                # conf_int usually has two columns [0,1] or named; take first and second
                ci_lower, ci_upper = float(row.iloc[0]), float(row.iloc[1])
            except Exception:
                # fallback: try dict-like access
                try:
                    ci_lower = float(conf_int['beauty_z'][0])
                    ci_upper = float(conf_int['beauty_z'][1])
                except Exception:
                    ci_lower = ci_upper = None

        # nobs if available
        nobs = None
        try:
            nobs = int(getattr(res, "nobs", None)) if getattr(res, "nobs", None) is not None else None
        except Exception:
            nobs = None

        # Convert numpy types to native python floats
        def _to_float(x):
            try:
                if x is None:
                    return None
                return float(x)
            except Exception:
                return None

        coef_f = _to_float(coef)
        std_err_f = _to_float(std_err)
        pval_f = _to_float(pval)
        ci_lower_f = _to_float(ci_lower)
        ci_upper_f = _to_float(ci_upper)

        sig = None
        if pval_f is not None:
            sig = (pval_f < 0.05)

        out.update({
            "coef": coef_f,
            "std_err": std_err_f,
            "p_value": pval_f,
            "ci_lower": ci_lower_f,
            "ci_upper": ci_upper_f,
            "nobs": nobs,
            "significant_0.05": bool(sig) if sig is not None else None
        })

        # Build description
        if coef_f is None:
            description = "Could not find a coefficient for 'beauty_z' in the provided model output."
        else:
            direction = "positive" if coef_f > 0 else ("zero" if coef_f == 0 else "negative")
            sig_text = ("statistically significant (p = {:.3g})".format(pval_f)
                        if pval_f is not None and pval_f < 0.05
                        else ("not statistically significant (p = {:.3g})".format(pval_f) if pval_f is not None
                              else "of unknown significance (p-value not found)"))
            ci_text = ("95% CI [{:.3f}, {:.3f}]".format(ci_lower_f, ci_upper_f)
                       if (ci_lower_f is not None and ci_upper_f is not None)
                       else "95% CI not available")
            nobs_text = (" based on {} observations".format(nobs) if nobs is not None else "")

            # Interpret coefficient: beauty_z is standardized, so coef = change in Eval per 1 SD in beauty
            description = (
                "Coefficient on beauty_z = {:.4f} (SE = {:.4f}). This is a {} effect: "
                "a one standard-deviation increase in instructor physical attractiveness is associated with a "
                "{:+.4f} point change in the course evaluation score (Eval). The effect is {}, {}{}. "
                "{}"
            ).format(
                coef_f,
                std_err_f if std_err_f is not None else float('nan'),
                direction,
                coef_f,
                sig_text,
                nobs_text,
                ci_text
            )

    except Exception as e:
        # If anything goes wrong, return an error description
        return {
            "object": None,
            "description": "Error extracting results from model_output: {}".format(repr(e))
        }

    return {"object": out, "description": description}