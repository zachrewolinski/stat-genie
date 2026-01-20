def extract_final_answer(model_output):
    """
    Extracts the effect of 'Children' from a fitted statsmodels GLM (Negative Binomial) result.
    
    Returns a dictionary with:
      - "object": dict with numeric results (coef, p-value, 95% CI, IRR, IRR 95% CI, percent change, significance boolean)
      - "description": human-readable interpretation in context (whether having children decreases extramarital affairs,
                       the strength/direction of the association, and statistical significance).
    """
    import numpy as np
    from math import exp

    result = {
        "coef": None,
        "pvalue": None,
        "ci_lower": None,
        "ci_upper": None,
        "IRR": None,
        "IRR_ci_lower": None,
        "IRR_ci_upper": None,
        "percent_change": None,
        "significant": None
    }

    try:
        # params, pvalues, conf_int as commonly provided by statsmodels result wrappers
        params = getattr(model_output, "params", None)
        pvalues = getattr(model_output, "pvalues", None)
        conf_int = None
        # conf_int may be a method or attribute
        if hasattr(model_output, "conf_int"):
            try:
                conf_int = model_output.conf_int()
            except TypeError:
                # if conf_int requires args, try without
                conf_int = model_output.conf_int()
        else:
            # try attribute
            conf_int = getattr(model_output, "conf_int_", None)

        # Extract for variable name 'Children' (exact match)
        if params is None or 'Children' not in params.index:
            raise KeyError("Could not find 'Children' in model parameters.")

        coef = float(params['Children'])
        pval = float(pvalues['Children']) if (pvalues is not None and 'Children' in pvalues.index) else np.nan

        # confidence interval on coef (log scale)
        ci_lower, ci_upper = (np.nan, np.nan)
        if conf_int is not None:
            try:
                # conf_int is typically a DataFrame or ndarray; handle both
                if hasattr(conf_int, "loc") and 'Children' in conf_int.index:
                    ci_lower = float(conf_int.loc['Children', 0])
                    ci_upper = float(conf_int.loc['Children', 1])
                else:
                    # assume array with same order as params.index
                    # find position of 'Children'
                    idx = list(params.index).index('Children')
                    ci_lower = float(conf_int[idx, 0])
                    ci_upper = float(conf_int[idx, 1])
            except Exception:
                ci_lower, ci_upper = (np.nan, np.nan)

        # Convert to incidence rate ratio (IRR) and CI on IRR scale
        irr = exp(coef)
        irr_ci_lower = exp(ci_lower) if not np.isnan(ci_lower) else np.nan
        irr_ci_upper = exp(ci_upper) if not np.isnan(ci_upper) else np.nan
        percent_change = (irr - 1.0) * 100.0  # percent change in expected count

        significant = bool((not np.isnan(pval)) and (pval < 0.05))

        result.update({
            "coef": coef,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "IRR": irr,
            "IRR_ci_lower": irr_ci_lower,
            "IRR_ci_upper": irr_ci_upper,
            "percent_change": percent_change,
            "significant": significant
        })

        # Build a concise description/interpretation
        if np.isnan(pval):
            sig_text = "p-value unavailable; cannot assess statistical significance."
        else:
            sig_text = ("statistically significant (p < 0.05)" if significant
                        else "not statistically significant (p >= 0.05)")

        direction = "decrease" if irr < 1 else ("increase" if irr > 1 else "no change")
        desc = (
            f"'Children' coefficient (log scale) = {coef:.4f}; 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
            f"Exponentiated IRR = {irr:.4f}; 95% CI for IRR = [{irr_ci_lower:.4f}, {irr_ci_upper:.4f}]. "
            f"This corresponds to a {percent_change:.1f}% expected {direction} in the reported frequency of "
            f"extramarital sexual intercourse for respondents with children versus without, holding controls constant. "
            f"The effect is {sig_text} (p = {pval:.4f})."
        )

        return {"object": result, "description": desc}

    except KeyError as e:
        return {
            "object": result,
            "description": f"Could not extract 'Children' coefficient from model_output: {e}"
        }
    except Exception as e:
        return {
            "object": result,
            "description": f"An error occurred while extracting results: {e}"
        }