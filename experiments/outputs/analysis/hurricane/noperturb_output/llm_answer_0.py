def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, p-value, and 95% CI for 'masfem_z' from available models
    in model_output. For Poisson/NegBin models, also returns the incidence-rate-ratio (IRR)
    and its 95% CI (exp(coef), exp(CI)).

    Returns a dict with keys:
      - "object": dict keyed by model name with extracted numeric results (or error messages)
      - "description": human-readable summary interpreting the direction and significance
                       of the 'masfem_z' coefficient in the context of the task.
    """
    import math
    import numpy as np

    available_models = ['ols_robust', 'poisson_robust', 'negbin_robust']
    extracted = {}
    summary_lines = []
    any_model_found = False
    any_successful_extraction = False

    for m in available_models:
        if m in model_output:
            any_model_found = True
            model = model_output[m]
            try:
                params = getattr(model, 'params', None)
                pvalues = getattr(model, 'pvalues', None)
                bse = getattr(model, 'bse', None)
                # conf_int may be a method or an attribute
                try:
                    ci_df = model.conf_int()
                except Exception:
                    ci_df = getattr(model, 'conf_int', None)
                    if callable(ci_df):
                        ci_df = ci_df()
                if params is None:
                    raise ValueError("model has no 'params' attribute")

                if 'masfem_z' not in params.index:
                    extracted[m] = {
                        "error": "'masfem_z' not present in model parameters",
                        "available_params": list(params.index)
                    }
                    summary_lines.append(f"{m}: 'masfem_z' not estimated in this model.")
                else:
                    coef = float(params.loc['masfem_z'])
                    se = float(bse.loc['masfem_z']) if (bse is not None and 'masfem_z' in bse.index) else None
                    pval = float(pvalues.loc['masfem_z']) if (pvalues is not None and 'masfem_z' in pvalues.index) else None
                    ci_low, ci_high = (None, None)
                    if ci_df is not None and 'masfem_z' in ci_df.index:
                        # conf_int returns DataFrame with two columns; ensure numeric
                        ci_low = float(ci_df.loc['masfem_z'].iloc[0])
                        ci_high = float(ci_df.loc['masfem_z'].iloc[1])

                    entry = {
                        "coef": coef,
                        "se": se,
                        "pvalue": pval,
                        "ci95": [ci_low, ci_high]
                    }

                    # For count models, provide exponentiated effect (IRR)
                    if m in ('poisson_robust', 'negbin_robust'):
                        try:
                            irr = float(np.exp(coef))
                            irr_ci = [float(np.exp(ci_low)) if ci_low is not None else None,
                                      float(np.exp(ci_high)) if ci_high is not None else None]
                        except Exception:
                            irr = None
                            irr_ci = [None, None]
                        entry.update({"irr": irr, "irr_ci95": irr_ci})

                    extracted[m] = entry

                    # Build summary line about direction and significance
                    sign = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
                    sig = ("statistically significant (p < 0.05)" if (pval is not None and pval < 0.05)
                           else ("not statistically significant" if pval is not None else "p-value unavailable"))
                    # Interpret in context: positive coef in OLS -> more-feminine -> more log deaths (less precaution)
                    if m == 'ols_robust':
                        interpretation = ("A positive coef implies more-feminine names are associated with higher "
                                          "log(1+deaths) (interpreted as less precaution).")
                    else:
                        interpretation = ("A positive coef implies more-feminine names are associated with a multiplicative "
                                          "increase in expected deaths; IRR > 1 indicates more deaths (less precaution).")
                    summary_lines.append(
                        f"{m}: coef={coef:.4g}, se={se:.4g} p={pval:.3g} 95%CI=[{ci_low:.4g}, {ci_high:.4g}] -> "
                        f"{sign}, {sig}. {interpretation}"
                    )
                    any_successful_extraction = True

            except Exception as e:
                extracted[m] = {"error": f"failed to extract fields from model object: {str(e)}"}
                summary_lines.append(f"{m}: extraction error: {str(e)}")
        else:
            # If model missing, try to surface error message if present
            err_key = m.replace('_robust', '_error')
            if err_key in model_output:
                msg = model_output[err_key]
                extracted[m] = {"error": msg}
                summary_lines.append(f"{m}: error reported when fitting model: {msg}")
            else:
                summary_lines.append(f"{m}: not present in model_output.")

    if not any_model_found:
        description = (
            "No fitted model objects ('ols_robust', 'poisson_robust', 'negbin_robust') were found in model_output. "
            "Model output keys present: " + ", ".join(list(model_output.keys())) + ". "
            "Common cause: pandas columns used as predictors/outcome are dtype 'object' (strings). "
            "Fix: ensure numeric columns (masfem_z, wind, min, category, elapsedyrs, log_ndam15, alldeaths) are "
            "converted to numeric (e.g., df[col] = pd.to_numeric(df[col], errors='coerce')) and refit models."
        )
        return {"object": extracted, "description": description}

    # Compose a compact description
    description = ""
    if any_successful_extraction:
        description = (
            "Extracted 'masfem_z' estimates from available models. Summary:\n" +
            "\n".join(summary_lines) +
            "\n\nInterpretation: For the OLS on log(1+deaths), a positive and statistically significant "
            "coefficient on 'masfem_z' would support the hypothesis that more-feminine hurricane names are "
            "associated with higher fatalities (consistent with less precaution), conditional on controls. "
            "For Poisson/NegBin, IRR > 1 (and significant) would similarly support the hypothesis."
        )
    else:
        # No successful extraction but some models existed
        description = (
            "Models were present but extraction failed or 'masfem_z' was not estimated. Details:\n"
            + "\n".join(summary_lines) +
            "\nCheck that 'masfem_z' exists in the dataframe used to fit models and that model objects "
            "are statsmodels result instances with attributes .params, .pvalues, .bse and method .conf_int()."
        )

    return {"object": extracted, "description": description}