def extract_final_answer(model_output):
    """
    Extract relevant statistics about the main predictor (masfem_z) or the binary
    gender predictor (gender_female) from a statsmodels-like result dict.

    Returns a dict with keys:
      - "object": a dict keyed by model name containing either extracted stats
                  or an error message when extraction is not possible.
      - "description": a short plain-English explanation of what was returned
                       and (if relevant) why extraction failed and how to fix it.
    """
    import numpy as np
    res = {}

    # Helper to safely extract stats for a given parameter name
    def _extract_from_result(obj, param):
        out = {}
        try:
            params = getattr(obj, "params", None)
            pvalues = getattr(obj, "pvalues", None)
            conf = None
            try:
                conf = obj.conf_int()
            except Exception:
                # some result-objects may require explicit alpha kw; catch all
                try:
                    conf = obj.conf_int(alpha=0.05)
                except Exception:
                    conf = None

            if params is None or param not in params.index:
                out["error"] = f"parameter '{param}' not found in result.params"
                return out

            coef = float(params[param])
            pval = float(pvalues[param]) if (pvalues is not None and param in pvalues.index) else None
            ci_low = ci_high = None
            if conf is not None and param in conf.index:
                ci_low, ci_high = float(conf.loc[param, 0]), float(conf.loc[param, 1])

            out["coef"] = coef
            out["pvalue"] = pval
            out["conf_int_95"] = (ci_low, ci_high)

            # If model is a count model (e.g., NegativeBinomial/GLM), report exp(coef)
            fam = None
            try:
                fam = getattr(getattr(obj, "model", None), "family", None)
            except Exception:
                fam = None
            is_count_family = False
            if fam is not None:
                fname = fam.__class__.__name__.lower()
                if "negativ" in fname or "poisson" in fname or "count" in fname:
                    is_count_family = True

            # Some wrappers (like robustcov_results) still expose model.family above;
            # if not found, we still provide exp(coef) because it is easy to interpret for counts.
            if is_count_family or True:
                try:
                    out["exp_coef"] = float(np.exp(coef))
                    if ci_low is not None and ci_high is not None:
                        out["exp_conf_int_95"] = (float(np.exp(ci_low)), float(np.exp(ci_high)))
                    else:
                        out["exp_conf_int_95"] = (None, None)
                except Exception:
                    out["exp_coef"] = None
                    out["exp_conf_int_95"] = (None, None)

            return out
        except Exception as e:
            return {"error": f"exception during extraction: {type(e).__name__}: {str(e)}"}

    # Validate input structure
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": f"Expected model_output to be a dict of models; got {type(model_output).__name__}."
        }

    # Iterate models and attempt extraction or record errors
    for name, obj in model_output.items():
        # If the entry is an Exception object (e.g., fitting failed), record the error string
        if isinstance(obj, BaseException):
            res[name] = {"error": str(obj)}
            continue

        # Try to extract masfem_z first; if not present, try gender_female
        try:
            params = getattr(obj, "params", None)
            if params is None:
                res[name] = {"error": "model object has no .params attribute; not a statsmodels result?"}
                continue
            idx = list(params.index)
            if "masfem_z" in idx:
                res[name] = _extract_from_result(obj, "masfem_z")
            elif "gender_female" in idx:
                res[name] = _extract_from_result(obj, "gender_female")
            else:
                # If neither parameter present, return available params for inspection
                res[name] = {
                    "warning": "neither 'masfem_z' nor 'gender_female' found in model parameters",
                    "available_params": idx
                }
        except Exception as e:
            res[name] = {"error": f"exception while parsing model result: {type(e).__name__}: {str(e)}"}

    # Compose user-facing description
    # If any model entries are errors, mention likely dtype problem (from the observed run)
    any_errors = any(("error" in v) for v in res.values())
    if any_errors:
        description = (
            "Extraction produced the per-model results (or errors) in 'object'. "
            "One or more models failed to fit or could not be parsed. If you saw "
            "TypeError(\"Cannot interpret 'Int64Dtype()' as a data type\"), this "
            "usually means pandas' nullable integer dtype (Int64) or other non-numpy "
            "dtypes were present; statsmodels expects native numpy dtypes. "
            "Fix by converting columns to native types before fitting, e.g.:\n"
            "  df = df.copy()\n"
            "  df['alldeaths_count'] = df['alldeaths_count'].astype(int)\n"
            "  df['masfem_z'] = df['masfem_z'].astype(float)\n"
            "  df['wind_z'] = df['wind_z'].astype(float)\n"
            "  df['min_z'] = df['min_z'].astype(float)\n"
            "  df['year_c'] = df['year_c'].astype(float)\n"
            "  df['gender_female'] = df['gender_female'].astype(int)\n"
            "  df['category'] = df['category'].astype('category')\n"
            "  df['source'] = df['source'].astype('category')\n"
            "Then re-run the models and call this extractor on the new results to obtain "
            "coefficients, p-values, confidence intervals, and (for count models) the "
            "exponentiated coefficients (incidence rate ratios)."
        )
    else:
        description = (
            "Extraction succeeded for all models. 'object' contains, per model, the "
            "coefficient, p-value, 95% CI for the target parameter (masfem_z or "
            "gender_female), and the exponentiated coefficient plus exponentiated CI "
            "(useful for interpreting count/Gamma/Poisson/NegativeBinomial models)."
        )

    return {"object": res, "description": description}