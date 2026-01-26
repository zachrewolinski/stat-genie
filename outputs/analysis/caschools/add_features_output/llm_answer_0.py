def extract_final_answer(model_output):
    """
    Extracts statistics for 'stu_teacher_ratio' from both unadjusted and adjusted models
    inside the provided model_output dict and returns a concise interpretation.

    Returns:
      {
        "object": {
          "unadjusted": { "coef": ..., "std_err": ..., "t": ..., "pvalue": ...,
                          "ci_lower": ..., "ci_upper": ..., "n": ... },
          "adjusted":   { ... same fields ... },
          "conclusion": "Yes / No / No evidence"  # based on adjusted model
        },
        "description": "Plain-language explanation of what the statistics mean"
      }
    """
    res = {"object": None, "description": None}

    # Basic checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing 'unadjusted' and 'adjusted' models.")

    # Helper to safely extract stats for a given fitted model
    def _extract_from_result(result, varname="stu_teacher_ratio"):
        stats = {}
        if result is None:
            return None
        # Try to extract core values; wrap in try/except to give useful errors if var missing
        try:
            params = result.params
            pvals = result.pvalues
            bse = result.bse
            tvalues = result.tvalues
            conf = result.conf_int()
        except Exception as e:
            raise RuntimeError(f"Unable to extract standard regression attributes from model result: {e}")

        if varname not in params.index:
            raise KeyError(f"Variable '{varname}' not found in model parameters. Available params: {list(params.index)}")

        coef = float(params[varname])
        se = float(bse[varname]) if varname in bse.index else float('nan')
        t = float(tvalues[varname]) if varname in tvalues.index else float('nan')
        p = float(pvals[varname]) if varname in pvals.index else float('nan')
        # conf_int may be a DataFrame/ndarray; try to access by label then fallback to positional
        try:
            ci = conf.loc[varname]
            ci_lower = float(ci.iloc[0])
            ci_upper = float(ci.iloc[1])
        except Exception:
            # fallback: find position of var in params.index
            try:
                idx = list(params.index).index(varname)
                ci_lower = float(conf[idx, 0])
                ci_upper = float(conf[idx, 1])
            except Exception:
                ci_lower = float('nan')
                ci_upper = float('nan')

        # sample size
        try:
            nobs = int(result.nobs)
        except Exception:
            # fallback: if model_output provides model_data_adjusted for adjusted model
            nobs = None

        stats.update({
            "coef": coef,
            "std_err": se,
            "t": t,
            "pvalue": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": nobs
        })
        return stats

    # Extract for unadjusted
    if 'unadjusted' not in model_output:
        raise KeyError("model_output missing key 'unadjusted'")
    if 'adjusted' not in model_output:
        raise KeyError("model_output missing key 'adjusted'")

    unadj_stats = _extract_from_result(model_output['unadjusted'], varname='stu_teacher_ratio')
    adj_stats = _extract_from_result(model_output['adjusted'], varname='stu_teacher_ratio')

    # If model_data_adjusted present, prefer its row count for adjusted n
    if 'model_data_adjusted' in model_output and hasattr(model_output['model_data_adjusted'], 'shape'):
        try:
            adj_stats['n'] = int(model_output['model_data_adjusted'].shape[0])
        except Exception:
            pass

    # Determine conclusion based on adjusted model
    conclusion = "No evidence (not statistically significant)"
    try:
        coef = adj_stats['coef']
        p = adj_stats['pvalue']
        if (p is not None) and (not (p != p)):  # check not NaN
            if p < 0.05:
                # significant: interpret sign
                if coef < 0:
                    conclusion = ("Yes: statistically significant. "
                                  "Lower student-teacher ratio (fewer students per teacher) is associated "
                                  "with higher academic performance (adjusted model).")
                else:
                    conclusion = ("No: statistically significant but in the opposite direction. "
                                  "Higher student-teacher ratio is associated with higher academic performance "
                                  "(adjusted model).")
            else:
                conclusion = "No evidence (estimate not statistically significant at alpha=0.05)."
        else:
            conclusion = "No conclusion: p-value is not available."
    except Exception:
        conclusion = "No conclusion: could not evaluate significance."

    # Prepare returned object
    result_object = {
        "unadjusted": unadj_stats,
        "adjusted": adj_stats,
        "conclusion": conclusion
    }

    # Short human-readable description
    # Interpret coefficient: it's change in AvgScore per one-unit increase in stu_teacher_ratio.
    # Therefore, a negative coef implies that decreasing the ratio (fewer students per teacher)
    # is associated with higher AvgScore; effect per 1-student decrease = -coef.
    if adj_stats is not None:
        coef = adj_stats.get('coef', float('nan'))
        p = adj_stats.get('pvalue', float('nan'))
        ci_low = adj_stats.get('ci_lower', float('nan'))
        ci_high = adj_stats.get('ci_upper', float('nan'))
        n = adj_stats.get('n', None)
        interp_lines = [
            f"Adjusted model (n={n}): coefficient on stu_teacher_ratio = {coef:.4f},",
            f"SE = {adj_stats.get('std_err', float('nan')):.4f}, p = {p:.4g},",
            f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]."
        ]
        if p < 0.05:
            if coef < 0:
                effect_sentence = (f"This implies that a one-student decrease in the student-teacher ratio "
                                   f"is associated with an average increase of {abs(coef):.4f} points in AvgScore, "
                                   f"holding controls constant. {conclusion}")
            else:
                effect_sentence = (f"This implies that a one-student decrease in the student-teacher ratio "
                                   f"is associated with an average decrease of {coef:.4f} points in AvgScore, "
                                   f"holding controls constant. {conclusion}")
        else:
            effect_sentence = ("The adjusted estimate is not statistically significant at the 0.05 level, "
                               "so we do not have sufficient evidence to conclude that student-teacher ratio is "
                               "associated with academic performance after controlling for the listed covariates.")
        interp_lines.append(effect_sentence)
        description = " ".join(interp_lines)
    else:
        description = "Could not extract adjusted-model statistics."

    res["object"] = result_object
    res["description"] = description
    return res