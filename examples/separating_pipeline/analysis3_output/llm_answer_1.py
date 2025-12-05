def extract_final_answer(model_output):
    """
    Extract statistics for the primary independent variable 'FemininityScore_z'
    from the provided model_output.

    Expects model_output to be a dict-like object with keys 'nb_model' and 'ols_model'
    (as returned by the modeling function). For each model it extracts:
      - coefficient
      - standard error
      - two-sided p-value
      - 95% confidence interval
    Additionally for the Negative Binomial (nb_model) it computes the incidence rate ratio (IRR)
    and its 95% CI (exp(coef) and exp(CI)).
    For the OLS on log(Fatalities + 1) it notes interpretation of the coefficient
    (change in log outcome; approximate percent change = 100*(exp(coef)-1)).

    The function returns a dict with keys:
      - "object": dict with extracted numeric results and a simple conclusion about
                  whether the evidence supports the hypothesis (coef < 0 and p < 0.05).
      - "description": human-readable explanation of the extracted numbers and interpretation.
    """
    import numpy as np

    # Helper to safely get models from possible input shapes
    nb_model = None
    ols_model = None
    if isinstance(model_output, dict):
        nb_model = model_output.get('nb_model', None)
        ols_model = model_output.get('ols_model', None)
    else:
        # try attribute access as fallback
        nb_model = getattr(model_output, 'nb_model', None)
        ols_model = getattr(model_output, 'ols_model', None)

    results = {}

    def extract_from_result(res, varname='FemininityScore_z'):
        """
        Extract coef, se, pvalue, conf_int for varname from a statsmodels results object.
        Returns dict or raises KeyError if varname not present.
        """
        # Some statsmodels wrappers expose these attributes as Series or DataFrame
        params = getattr(res, 'params', None)
        pvalues = getattr(res, 'pvalues', None)
        bse = getattr(res, 'bse', None)
        try:
            ci = res.conf_int()
        except Exception:
            ci = None

        if params is None or varname not in params.index:
            raise KeyError(f"Variable '{varname}' not found in result params.")

        coef = float(params[varname])
        se = float(bse[varname]) if (bse is not None and varname in bse.index) else None
        pval = float(pvalues[varname]) if (pvalues is not None and varname in pvalues.index) else None
        if ci is not None and varname in ci.index:
            ci_low = float(ci.loc[varname][0])
            ci_high = float(ci.loc[varname][1])
        else:
            ci_low = ci_high = None

        return {"coef": coef, "se": se, "pvalue": pval, "ci_lower": ci_low, "ci_upper": ci_high}

    # Extract for Negative Binomial model if present
    if nb_model is not None:
        try:
            nb_stats = extract_from_result(nb_model, 'FemininityScore_z')
            # For count model, transform coef to incidence rate ratio
            irr = float(np.exp(nb_stats['coef']))
            irr_ci_lower = float(np.exp(nb_stats['ci_lower'])) if nb_stats['ci_lower'] is not None else None
            irr_ci_upper = float(np.exp(nb_stats['ci_upper'])) if nb_stats['ci_upper'] is not None else None

            nb_stats.update({"IRR": irr, "IRR_ci_lower": irr_ci_lower, "IRR_ci_upper": irr_ci_upper})
            # simple decision: negative coef and statistically significant (two-sided p < .05)
            nb_stats["supports_hypothesis"] = (nb_stats["coef"] < 0) and (nb_stats["pvalue"] is not None and nb_stats["pvalue"] < 0.05)

            results['nb_model'] = nb_stats
        except KeyError as e:
            results['nb_model'] = {"error": str(e)}
        except Exception as e:
            results['nb_model'] = {"error": f"Unexpected error extracting nb_model stats: {e}"}
    else:
        results['nb_model'] = None

    # Extract for OLS model if present
    if ols_model is not None:
        try:
            ols_stats = extract_from_result(ols_model, 'FemininityScore_z')
            # Interpret coefficient on log(Fatalities + 1): approximate percent change = 100*(exp(coef)-1)
            try:
                pct_change_approx = 100.0 * (np.exp(ols_stats['coef']) - 1.0)
            except Exception:
                pct_change_approx = None
            ols_stats.update({"approx_pct_change_in_fatalities_plus1": pct_change_approx})
            ols_stats["supports_hypothesis"] = (ols_stats["coef"] < 0) and (ols_stats["pvalue"] is not None and ols_stats["pvalue"] < 0.05)

            results['ols_model'] = ols_stats
        except KeyError as e:
            results['ols_model'] = {"error": str(e)}
        except Exception as e:
            results['ols_model'] = {"error": f"Unexpected error extracting ols_model stats: {e}"}
    else:
        results['ols_model'] = None

    # Build a short textual conclusion combining both models
    conclusions = []
    for name in ['nb_model', 'ols_model']:
        info = results.get(name)
        if info is None:
            conclusions.append(f"{name}: model not provided.")
        elif "error" in info:
            conclusions.append(f"{name}: {info['error']}")
        else:
            sign = "negative" if info['coef'] < 0 else "positive" if info['coef'] > 0 else "zero"
            sig = "significant (p < 0.05)" if info.get('pvalue') is not None and info['pvalue'] < 0.05 else f"not significant (p = {info.get('pvalue'):.3g})" if info.get('pvalue') is not None else "p-value unavailable"
            if name == 'nb_model':
                conclusions.append(
                    f"NB: coef={info['coef']:.4f} (SE={info['se']:.4f}), IRR={info['IRR']:.3f}, "
                    f"95%CI_IRR=[{info['IRR_ci_lower']:.3f}, {info['IRR_ci_upper']:.3f}] -> {sign}, {sig}."
                )
            else:
                pct = info.get('approx_pct_change_in_fatalities_plus1')
                pct_text = f"approx {pct:.2f}% change in (Fatalities+1)" if pct is not None else "percent change unavailable"
                conclusions.append(
                    f"OLS on log(Fatalities+1): coef={info['coef']:.4f} (SE={info['se']:.4f}), "
                    f"95%CI=[{info['ci_lower']:.4f}, {info['ci_upper']:.4f}] -> {sign}, {sig}; {pct_text}."
                )

    # Overall combined judgement: if both models that ran support hypothesis, flag highly consistent.
    support_flags = []
    for name in ['nb_model', 'ols_model']:
        info = results.get(name)
        if info and isinstance(info, dict) and 'supports_hypothesis' in info:
            support_flags.append(bool(info['supports_hypothesis']))
    if len(support_flags) == 0:
        overall = "No model results available to form a conclusion."
    elif all(support_flags):
        overall = "Both models' FemininityScore_z coefficients are negative and statistically significant -> evidence supports the hypothesis."
    elif any(support_flags):
        overall = "At least one model shows a negative, statistically significant association -> partial support for the hypothesis."
    else:
        overall = "Neither model shows a negative, statistically significant association -> no evidence supporting the hypothesis."

    description = (
        "Extracted statistics for 'FemininityScore_z' from Negative Binomial and OLS robustness models.\n"
        "For the Negative Binomial model, coefficients are on the log-count scale; IRR = exp(coef) is reported.\n"
        "For the OLS on log(Fatalities + 1), coefficient is the change in log outcome; "
        "approx percent change in (Fatalities+1) is 100*(exp(coef)-1).\n\n"
        "Summary of model-specific results:\n" + "\n".join(conclusions) + "\n\n"
        "Overall assessment: " + overall
    )

    return {"object": results, "description": description}