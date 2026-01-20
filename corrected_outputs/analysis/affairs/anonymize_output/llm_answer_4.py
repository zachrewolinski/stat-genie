def extract_final_answer(model_output):
    """
    Extracts statistics for the 'HasChildren' variable from a fitted
    statsmodels ZeroInflatedNegativeBinomialResultsWrapper object.

    Returns a dict with:
      - "object": a dictionary of numeric results for the HasChildren effect
                  in both the count and zero-inflation parts
      - "description": a short, plain-language interpretation of those results
    """
    import numpy as np
    import pandas as pd

    # Helper to get parameter table (params, bse, pvalues, conf_int) in a consistent form
    # Attempt to use result attributes (most statsmodels results provide these)
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        conf_int = model_output.conf_int()
    except Exception:
        # If any attribute missing, raise informative error
        raise ValueError("Provided model_output does not expose expected attributes "
                         "(params, bse, pvalues, conf_int). Provide a statsmodels results object.")

    # Ensure params is a pandas Series so we can index by name; if it's ndarray, fall back to model metadata
    if not isinstance(params, pd.Series):
        # try to recover names from model object
        exog_names = getattr(model_output.model, "exog_names", None)
        exog_infl_names = getattr(model_output.model, "exog_infl_names", None)
        if exog_names is None or exog_infl_names is None:
            raise ValueError("Cannot determine parameter names from model_output.")
        # build index
        names = list(exog_names) + [f"inflate_{n}" for n in exog_infl_names] + ["alpha"]
        params = pd.Series(np.asarray(params), index=names)
        bse = pd.Series(np.asarray(bse), index=names)
        pvalues = pd.Series(np.asarray(pvalues), index=names)
        # conf_int may be ndarray
        if isinstance(conf_int, (list, np.ndarray)):
            conf_int = pd.DataFrame(conf_int, index=names)
        else:
            conf_int = pd.DataFrame(conf_int, index=names)

    # Normalize index to simple strings
    param_names = list(params.index.astype(str))

    # Find parameter name(s) corresponding to HasChildren in both parts.
    # Typical names:
    #  - count part: 'HasChildren'
    #  - inflation part: 'inflate_HasChildren' or contains 'HasChildren' and 'infl' in name
    count_name = None
    infl_name = None

    # direct match
    if "HasChildren" in param_names:
        count_name = "HasChildren"

    # inflation candidate direct
    for n in param_names:
        ln = n.lower()
        if "haschildren" in ln and ("infl" in ln or "inflate" in ln):
            infl_name = n
            break

    # if not found by above, search more heuristically:
    if count_name is None:
        # choose name containing HasChildren but not inflation-related
        for n in param_names:
            if "haschildren" in n.lower() and not any(k in n.lower() for k in ["infl", "inflate"]):
                count_name = n
                break

    if infl_name is None:
        for n in param_names:
            if "haschildren" in n.lower() and any(k in n.lower() for k in ["infl", "inflate"]):
                infl_name = n
                break

    # If still not found, try matching by position: assume exog_names precede inflation names
    if (count_name is None or infl_name is None):
        exog_names = getattr(model_output.model, "exog_names", None)
        exog_infl_names = getattr(model_output.model, "exog_infl_names", None)
        if exog_names is not None and exog_infl_names is not None:
            # if exog_names has HasChildren
            if count_name is None and "HasChildren" in exog_names:
                count_name = "HasChildren"
            if infl_name is None and "HasChildren" in exog_infl_names:
                # find actual inflation param name (likely prefixed in params)
                # look for any param name that contains the exog_infl name
                target = "haschildren"
                for n in param_names:
                    if target in n.lower() and any(k in n.lower() for k in ["infl", "inflate"]):
                        infl_name = n
                        break

    # Final safety checks
    if count_name is None and infl_name is None:
        raise ValueError("Could not locate any parameter corresponding to 'HasChildren' in model_output.")

    results = {}

    # Extract count part results if available
    if count_name is not None and count_name in params.index:
        coef = float(params[count_name])
        se = float(bse[count_name]) if count_name in bse.index else None
        pval = float(pvalues[count_name]) if count_name in pvalues.index else None
        ci_low, ci_high = None, None
        if count_name in conf_int.index:
            ci_row = conf_int.loc[count_name]
            ci_low, ci_high = float(ci_row.iloc[0]), float(ci_row.iloc[1])
        else:
            # approximate
            if se is not None:
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se

        irr = float(np.exp(coef))
        irr_ci = (float(np.exp(ci_low)), float(np.exp(ci_high))) if (ci_low is not None and ci_high is not None) else (None, None)

        results['count'] = {
            "param_name": count_name,
            "coef": coef,
            "std_err": se,
            "p_value": pval,
            "coef_95ci": (ci_low, ci_high),
            "incidence_rate_ratio": irr,
            "irr_95ci": irr_ci,
            "interpretation_brief": (
                "Count model: coefficient is on the log expected-count scale. "
                "IRR < 1 means HasChildren is associated with lower expected affair frequency "
                "among those in the count process."
            )
        }

    # Extract inflation part results if available
    if infl_name is not None and infl_name in params.index:
        coef = float(params[infl_name])
        se = float(bse[infl_name]) if infl_name in bse.index else None
        pval = float(pvalues[infl_name]) if infl_name in pvalues.index else None
        ci_low, ci_high = None, None
        if infl_name in conf_int.index:
            ci_row = conf_int.loc[infl_name]
            ci_low, ci_high = float(ci_row.iloc[0]), float(ci_row.iloc[1])
        else:
            if se is not None:
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se

        odds_ratio = float(np.exp(coef))
        or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high))) if (ci_low is not None and ci_high is not None) else (None, None)

        results['inflation'] = {
            "param_name": infl_name,
            "coef": coef,
            "std_err": se,
            "p_value": pval,
            "coef_95ci": (ci_low, ci_high),
            "odds_ratio_for_structural_zero": odds_ratio,
            "or_95ci": or_ci,
            "interpretation_brief": (
                "Inflation (logit) model: coefficient is the log-odds of being a structural (always) zero. "
                "OR > 1 means HasChildren increases the odds of being in the always-zero group "
                "(i.e., being in the group that never has affairs)."
            )
        }

    # Synthesis: give concise conclusion about whether having children decreases engagement
    synth_lines = []
    # Evaluate significance using conventional alpha=0.05 if p-values present
    alpha = 0.05
    if 'count' in results:
        c = results['count']
        sig = (c['p_value'] is not None and c['p_value'] < alpha)
        direction = "decrease" if c['incidence_rate_ratio'] < 1 else ("increase" if c['incidence_rate_ratio'] > 1 else "no change")
        synth_lines.append(
            f"Count part: coef={c['coef']:.4f}, IRR={c['incidence_rate_ratio']:.4f} "
            f"(95%CI IRR=({c['irr_95ci'][0]:.4f}, {c['irr_95ci'][1]:.4f}) if available), p={c['p_value']:.4g}. "
            f"Interpreted as a {direction} in expected affair frequency; "
            f"{'statistically significant' if sig else 'not statistically significant'}."
        )
    if 'inflation' in results:
        i = results['inflation']
        sig = (i['p_value'] is not None and i['p_value'] < alpha)
        direction = "increase" if i['odds_ratio_for_structural_zero'] > 1 else ("decrease" if i['odds_ratio_for_structural_zero'] < 1 else "no change")
        synth_lines.append(
            f"Inflation part: coef={i['coef']:.4f}, OR={i['odds_ratio_for_structural_zero']:.4f} "
            f"(95%CI OR=({i['or_95ci'][0]:.4f}, {i['or_95ci'][1]:.4f}) if available), p={i['p_value']:.4g}. "
            f"Interpreted as a {direction} in the odds of being a structural non-participant (no affairs); "
            f"{'statistically significant' if sig else 'not statistically significant'}."
        )

    # Final plain-language summary
    if 'count' in results and 'inflation' in results:
        # If both point to reductions and at least one significant, say children are associated with reduced engagement.
        count_ir = results['count']['incidence_rate_ratio']
        infl_or = results['inflation']['odds_ratio_for_structural_zero']
        count_sig = (results['count']['p_value'] is not None and results['count']['p_value'] < alpha)
        infl_sig = (results['inflation']['p_value'] is not None and results['inflation']['p_value'] < alpha)

        if ((count_ir < 1 and count_sig) or (infl_or > 1 and infl_sig)) or (count_ir < 1 and infl_or > 1):
            final_conclusion = ("Overall: Having children is associated with lower engagement in extramarital affairs. "
                                "This appears as (a) a lower expected frequency among those in the count process (IRR < 1) "
                                "and/or (b) higher odds of being a structural non-participant (OR > 1). "
                                "Check p-values above to see which effects are statistically significant.")
        else:
            final_conclusion = ("Overall: The point estimates suggest having children is associated with lower engagement "
                                "in affairs (lower count IRR and/or higher inflation OR), but the evidence is not strongly "
                                "statistically significant for both parts. Inspect p-values and CIs above for details.")
    else:
        # Only one part available
        if 'count' in results:
            ci = results['count']
            if ci['p_value'] is not None and ci['p_value'] < alpha and ci['incidence_rate_ratio'] < 1:
                final_conclusion = ("Overall: Having children is associated with a statistically significant reduction "
                                    "in expected affair frequency (count model).")
            else:
                final_conclusion = ("Overall: The count-model estimate suggests children lower expected affair frequency "
                                    "(IRR < 1), but it is not clearly statistically significant.")
        elif 'inflation' in results:
            ii = results['inflation']
            if ii['p_value'] is not None and ii['p_value'] < alpha and ii['odds_ratio_for_structural_zero'] > 1:
                final_conclusion = ("Overall: Having children is associated with statistically significantly higher odds "
                                    "of being a structural non-participant (i.e., never having affairs).")
            else:
                final_conclusion = ("Overall: The inflation-model estimate suggests children raise the odds of being "
                                    "a structural non-participant (reducing likelihood of any affair), but it is not clearly "
                                    "statistically significant.")

    description = (
        "Extracted coefficients, standard errors, p-values, and 95% confidence intervals for 'HasChildren' "
        "from both the count and zero-inflation parts of the fitted ZINB model. Interpretation: "
        "In the count part, coefficients are on the log expected-count scale; exponentiated coefficients "
        "give incidence rate ratios (IRR). In the zero-inflation (logit) part, coefficients give log-odds of "
        "being a structural zero; exponentiated coefficients give odds ratios (OR) for being always-zero. "
        "A negative (and significant) count coef (IRR < 1) and/or a positive (and significant) inflation coef (OR > 1) "
        "indicate that having children is associated with decreased engagement in extramarital affairs. "
        "Summary: " + final_conclusion
    )

    return {"object": results, "description": description}