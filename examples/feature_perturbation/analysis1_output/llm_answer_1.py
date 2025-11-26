def extract_final_answer(model_output):
    """
    Extract statistics for the 'masfem_z' predictor from a fitted statsmodels GLMResultsWrapper
    (Negative Binomial) and provide an interpretation relevant to the hypothesis.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coef, se, z, pvalue, conf_int, IRR, IRR_conf_int, significant, nobs)
      - "description": a brief plain-language interpretation of those statistics in context.
    """
    import numpy as np
    res = model_output

    var = 'masfem_z'
    # Prepare container
    result_obj = {}
    try:
        # Coefficient, SE, z (or t), p-value
        coef = float(res.params[var])
        se = float(res.bse[var])
        # GLM provides tvalues (or z/t); statsmodels names this attribute tvalues
        stat = float(res.tvalues[var]) if var in res.tvalues.index else float(res.tvalues[list(res.params.index).index(var)])
        pval = float(res.pvalues[var])

        # Confidence interval for the coefficient
        try:
            ci = res.conf_int().loc[var].astype(float)  # pandas Series expected
            ci_low, ci_high = float(ci.iloc[0]), float(ci.iloc[1])
        except Exception:
            # fallback: conf_int as ndarray, find index of var
            ci_array = res.conf_int()
            try:
                idx = list(res.params.index).index(var)
                ci_low, ci_high = float(ci_array[idx, 0]), float(ci_array[idx, 1])
            except Exception:
                ci_low, ci_high = (None, None)

        # Incidence Rate Ratio (IRR) and its CI: exp(coef)
        irr = float(np.exp(coef))
        irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

        # Sample size if available
        nobs = int(res.nobs) if hasattr(res, 'nobs') else None

        # Significance at alpha=0.05
        significant = (pval < 0.05)

        # Pack numeric results
        result_obj = {
            "variable": var,
            "coef_log_count": coef,
            "std_error": se,
            "statistic": stat,
            "p_value": pval,
            "conf_int_95": [ci_low, ci_high],
            "IRR": irr,
            "IRR_conf_int_95": [irr_ci_low, irr_ci_high],
            "significant_at_0.05": bool(significant),
            "nobs": nobs
        }

        # Interpret the direction relative to hypothesis:
        # Hypothesis: more feminine names -> fewer precautions -> more fatalities.
        if significant:
            if coef > 0:
                conclusion = (
                    "The coefficient for masfem_z is positive and statistically significant (p < 0.05). "
                    "That implies higher femininity (one SD increase) is associated with higher expected "
                    "fatalities. In multiplicative terms, the expected death count is multiplied by ≈{:.3f} "
                    "(95% CI: {:.3f} to {:.3f}) per one SD increase in name femininity. "
                    "This result is consistent with the hypothesis that more feminine hurricane names lead "
                    "to fewer precautions and thus more fatalities."
                ).format(irr, irr_ci_low if irr_ci_low is not None else float('nan'),
                         irr_ci_high if irr_ci_high is not None else float('nan'))
            else:
                conclusion = (
                    "The coefficient for masfem_z is negative and statistically significant (p < 0.05). "
                    "That implies higher femininity (one SD increase) is associated with lower expected "
                    "fatalities. In multiplicative terms, the expected death count is multiplied by ≈{:.3f} "
                    "(95% CI: {:.3f} to {:.3f}) per one SD increase in name femininity. "
                    "This result is contrary to the hypothesis (it would suggest more feminine names lead to "
                    "fewer fatalities)."
                ).format(irr, irr_ci_low if irr_ci_low is not None else float('nan'),
                         irr_ci_high if irr_ci_high is not None else float('nan'))
        else:
            # Not statistically significant
            direction = "positive" if coef > 0 else "negative" if coef < 0 else "near zero"
            conclusion = (
                "The coefficient for masfem_z is {} but not statistically significant (p = {:.3g}). "
                "Point estimate: log-count coef = {:.4f}, IRR ≈ {:.3f} (95% CI: {:.3f} to {:.3f}). "
                "Because the effect is not statistically significant, we cannot conclude evidence for or against "
                "the hypothesis based on this model."
            ).format(direction, pval, coef, irr,
                     irr_ci_low if irr_ci_low is not None else float('nan'),
                     irr_ci_high if irr_ci_high is not None else float('nan'))

        description = (
            "Extracted the coefficient and inference for 'masfem_z' from a Negative Binomial GLM predicting "
            "hurricane deaths. 'coef_log_count' is the estimated change in log expected deaths per one SD "
            "increase in name femininity. 'IRR' = exp(coef) is the multiplicative change in expected deaths. "
            "p_value and conf_int_95 provide inferential information. Conclusion: " + conclusion
        )

        return {"object": result_obj, "description": description}

    except KeyError:
        # Variable not found in model
        return {
            "object": None,
            "description": f"The fitted model does not contain a parameter named '{var}'. "
                           "Ensure the model included 'masfem_z' as a predictor."
        }
    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting results for '{var}': {repr(e)}"
        }