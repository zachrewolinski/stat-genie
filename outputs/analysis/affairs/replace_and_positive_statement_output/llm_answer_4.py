def extract_final_answer(model_output):
    """
    Extracts the effect of 'children_yes' from:
      - the logistic regression ("logit") predicting any affair
      - the Zero-Inflated Negative Binomial ("zinb") on count of affairs

    Returns a dict:
      - "object": structured numeric results for the relevant coefficients
      - "description": concise interpretation in plain language

    Notes:
      - Expects model_output to be the dict returned by the provided model() function,
        with keys 'logit' and 'zinb' whose values are statsmodels result wrappers.
    """
    import math
    from collections import OrderedDict

    out = OrderedDict()
    try:
        logit_res = model_output['logit']
        zinb_res = model_output['zinb']
    except Exception as e:
        return {
            "object": None,
            "description": f"Input model_output must be a dict with keys 'logit' and 'zinb'. Error: {e}"
        }

    def safe_get(res, name):
        """Return coef, se, pval, ci (list [low, high]) for a parameter name if present, else None."""
        try:
            params = res.params
            pvals = getattr(res, 'pvalues', None)
            ses = getattr(res, 'bse', None)
            ci_df = None
            try:
                ci_df = res.conf_int()
            except Exception:
                ci_df = None

            if name in params.index:
                coef = float(params[name])
                se = float(ses[name]) if ses is not None and name in ses.index else None
                pval = float(pvals[name]) if pvals is not None and name in pvals.index else None
                ci = [float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])] if (ci_df is not None and name in ci_df.index) else [None, None]
                return {"coef": coef, "se": se, "pval": pval, "ci": ci, "param_name": name}
            # try to find a parameter whose name ends with the provided name (robustness for prefixed inflation names)
            for idx in params.index:
                if idx.endswith(name):
                    coef = float(params[idx])
                    se = float(ses[idx]) if ses is not None and idx in ses.index else None
                    pval = float(pvals[idx]) if pvals is not None and idx in pvals.index else None
                    ci = [float(ci_df.loc[idx, 0]), float(ci_df.loc[idx, 1])] if (ci_df is not None and idx in ci_df.index) else [None, None]
                    return {"coef": coef, "se": se, "pval": pval, "ci": ci, "param_name": idx}
        except Exception:
            pass
        return None

    # 1) Extract from logistic model (predicting any affair)
    logit_children = safe_get(logit_res, 'children_yes')
    logit_summary = None
    if logit_children is not None:
        coef = logit_children['coef']
        pval = logit_children['pval']
        ci = logit_children['ci']
        orr = math.exp(coef)
        or_ci = [math.exp(ci[0]) if ci[0] is not None else None, math.exp(ci[1]) if ci[1] is not None else None]
        logit_summary = {
            "param_name": logit_children["param_name"],
            "coef_logit": coef,
            "se": logit_children["se"],
            "pval": pval,
            "ci_logit": ci,
            "odds_ratio": orr,
            "odds_ratio_ci": or_ci
        }
    else:
        logit_summary = {"error": "children_yes not found in logit model parameters."}

    out['logit'] = logit_summary

    # 2) Extract from ZINB model.
    # We aim to report:
    #   - count part coefficient for children_yes (effect on expected count among the count-process)
    #   - inflation part coefficient for children_yes (effect on odds of being an 'excess' zero)
    zinb_summary = {}
    # Try to get count-part parameter: the count-part parameter names are usually present as plain names
    count_part = safe_get(zinb_res, 'children_yes')
    if count_part is not None:
        coef = count_part['coef']
        pval = count_part['pval']
        ci = count_part['ci']
        irr = math.exp(coef)  # incidence rate ratio
        irr_ci = [math.exp(ci[0]) if ci[0] is not None else None, math.exp(ci[1]) if ci[1] is not None else None]
        zinb_summary['count_part'] = {
            "param_name": count_part["param_name"],
            "coef_count": coef,
            "se": count_part['se'],
            "pval": pval,
            "ci_count": ci,
            "incidence_rate_ratio": irr,
            "irr_ci": irr_ci,
            "interpretation": ("IRR < 1 means having children is associated with a lower expected number of affairs "
                               "among the population not belonging to the 'always-zero' group.")
        }
    else:
        zinb_summary['count_part'] = {"error": "children_yes not found in ZINB count-part parameters."}

    # Inflation part - look for prefixed/inflated parameter; safe_get will find by suffix match if necessary.
    # Zero-inflation params are often named like 'inflate_children_yes' in results; safe_get handles suffix match.
    infl_part = safe_get(zinb_res, 'children_yes')  # safe_get will find first matching param (could be count); need to search inflation explicitly
    # To be more robust, inspect parameter names to prioritize inflation parameter if present
    infl_param_name = None
    try:
        param_index = list(zinb_res.params.index)
        # look for explicit inflation-prefixed names
        candidates = [p for p in param_index if ('inflate' in p.lower() or 'infl' in p.lower()) and p.endswith('children_yes')]
        if len(candidates) == 1:
            infl_param_name = candidates[0]
        else:
            # fallback: find any param that endswith children_yes but is not identical to the count-part name we already used
            for p in param_index:
                if p.endswith('children_yes'):
                    # if we've already used this name for count part, try to find another one; otherwise accept it
                    if ('count_part' in zinb_summary and isinstance(zinb_summary['count_part'], dict)
                        and 'param_name' in zinb_summary['count_part'] and p == zinb_summary['count_part']['param_name']):
                        continue
                    infl_param_name = p
                    break
    except Exception:
        infl_param_name = None

    infl_summary = None
    if infl_param_name is not None:
        # get values
        try:
            params = zinb_res.params
            pvals = getattr(zinb_res, 'pvalues', None)
            ses = getattr(zinb_res, 'bse', None)
            ci_df = None
            try:
                ci_df = zinb_res.conf_int()
            except Exception:
                ci_df = None
            coef = float(params[infl_param_name])
            se = float(ses[infl_param_name]) if ses is not None and infl_param_name in ses.index else None
            pval = float(pvals[infl_param_name]) if pvals is not None and infl_param_name in pvals.index else None
            ci = [float(ci_df.loc[infl_param_name, 0]), float(ci_df.loc[infl_param_name, 1])] if (ci_df is not None and infl_param_name in ci_df.index) else [None, None]
            or_infl = math.exp(coef)  # effect on odds of being an excessive zero
            infl_summary = {
                "param_name": infl_param_name,
                "coef_infl": coef,
                "se": se,
                "pval": pval,
                "ci_infl": ci,
                "odds_ratio_inflation": or_infl,
                "or_inflation_ci": [math.exp(ci[0]) if ci[0] is not None else None, math.exp(ci[1]) if ci[1] is not None else None],
                "interpretation": ("Positive inflation coef -> higher odds of being an 'excess' zero (i.e., more likely to be always-zero). "
                                   "Negative inflation coef -> lower odds of being an excess zero.")
            }
        except Exception:
            infl_summary = {"error": "Failed to extract inflation parameter values despite detecting a name."}
    else:
        # If we didn't detect any inflation-specific param name, attempt a secondary more permissive search:
        infl_summary = {"note": "No separate inflation parameter for 'children_yes' detected (or name ambiguous)."}
    zinb_summary['inflation_part'] = infl_summary

    out['zinb'] = zinb_summary

    # Build an overall interpretation string using available p-values and directions.
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    interpretation_lines = []
    # evaluate logit evidence
    try:
        if isinstance(logit_summary, dict) and 'coef_logit' in logit_summary:
            coef = logit_summary['coef_logit']
            p = logit_summary['pval']
            orr = logit_summary['odds_ratio']
            or_low, or_high = logit_summary['odds_ratio_ci']
            interpretation_lines.append(
                "Logistic model (any affair): children_yes coef = {coef}, OR = {or_} (95% CI {low} to {high}), p = {p}."
                .format(coef=fmt(coef), or_=fmt(orr), low=(fmt(or_low) if or_low is not None else None),
                        high=(fmt(or_high) if or_high is not None else None), p=(fmt(p) if p is not None else None))
            )
            if p is not None and p < 0.05:
                if orr < 1:
                    interpretation_lines.append("Interpretation: Having children is associated with a statistically significant decrease in the odds of any extramarital affair.")
                else:
                    interpretation_lines.append("Interpretation: Having children is associated with a statistically significant increase in the odds of any extramarital affair.")
            else:
                interpretation_lines.append("Interpretation: Logistic model does not show a statistically significant association between having children and the probability of any affair (at alpha=0.05).")
        else:
            interpretation_lines.append("Logistic model: children_yes parameter not available.")
    except Exception:
        interpretation_lines.append("Could not interpret logistic model output for children_yes.")

    # evaluate zinb evidence
    try:
        cp = zinb_summary.get('count_part', {})
        ip = zinb_summary.get('inflation_part', {})
        if isinstance(cp, dict) and 'coef_count' in cp:
            coef = cp['coef_count']
            p = cp['pval']
            irr = cp['incidence_rate_ratio']
            irr_low, irr_high = cp['irr_ci']
            interpretation_lines.append(
                "ZINB count part: children_yes coef = {coef}, IRR = {irr} (95% CI {low} to {high}), p = {p}."
                .format(coef=fmt(coef), irr=fmt(irr),
                        low=(fmt(irr_low) if irr_low is not None else None),
                        high=(fmt(irr_high) if irr_high is not None else None),
                        p=(fmt(p) if p is not None else None))
            )
            if p is not None and p < 0.05:
                if irr < 1:
                    interpretation_lines.append("Interpretation: Among the 'at-risk' (count) process, having children is associated with a statistically significant lower expected number of affairs.")
                else:
                    interpretation_lines.append("Interpretation: Among the 'at-risk' (count) process, having children is associated with a statistically significant higher expected number of affairs.")
            else:
                interpretation_lines.append("Interpretation: ZINB count part does not show a statistically significant association between children and the expected number of affairs (at alpha=0.05).")
        else:
            interpretation_lines.append("ZINB count part: children_yes parameter not available.")
        # inflation part interpretation
        if isinstance(ip, dict) and 'coef_infl' in ip:
            coef = ip['coef_infl']
            p = ip['pval']
            or_infl = ip['odds_ratio_inflation']
            or_low, or_high = ip['or_inflation_ci']
            interpretation_lines.append(
                "ZINB inflation part: children_yes coef = {coef}, OR_infl = {or_} (95% CI {low} to {high}), p = {p}."
                .format(coef=fmt(coef), or_=fmt(or_infl),
                        low=(fmt(or_low) if or_low is not None else None),
                        high=(fmt(or_high) if or_high is not None else None),
                        p=(fmt(p) if p is not None else None))
            )
            if p is not None and p < 0.05:
                if or_infl > 1:
                    interpretation_lines.append("Interpretation: Having children significantly increases the odds of being in the 'always-zero' group (consistent with reduced engagement).")
                else:
                    interpretation_lines.append("Interpretation: Having children significantly decreases the odds of being in the 'always-zero' group (consistent with increased engagement).")
            else:
                interpretation_lines.append("Interpretation: ZINB inflation part does not show a statistically significant association (at alpha=0.05).")
        else:
            # if no inflation param found, include note
            if isinstance(ip, dict) and 'note' in ip:
                interpretation_lines.append("ZINB inflation part: " + ip['note'])
            else:
                interpretation_lines.append("ZINB inflation part: children_yes parameter not available or ambiguous.")
    except Exception:
        interpretation_lines.append("Could not interpret ZINB model output for children_yes.")

    # Combine decision: consider evidence supportive if either logistic OR ZINB count shows significant association in direction of decrease
    decision = "Inconclusive based on available outputs."
    try:
        sig_decrease = False
        sig_increase = False
        # logistic
        if isinstance(logit_summary, dict) and logit_summary.get('pval') is not None:
            p = logit_summary['pval']
            if p < 0.05:
                if logit_summary['odds_ratio'] < 1:
                    sig_decrease = True
                else:
                    sig_increase = True
        # zainb count
        cp = zinb_summary.get('count_part', {})
        if isinstance(cp, dict) and cp.get('pval') is not None:
            p = cp['pval']
            if p < 0.05:
                if cp['incidence_rate_ratio'] < 1:
                    sig_decrease = True
                else:
                    sig_increase = True
        # consider inflation part consistent evidence if significant and indicates more always-zeros (OR_infl >1)
        ip = zinb_summary.get('inflation_part', {})
        if isinstance(ip, dict) and ip.get('pval') is not None and ip.get('odds_ratio_inflation') is not None:
            if ip['pval'] < 0.05:
                if ip['odds_ratio_inflation'] > 1:
                    sig_decrease = True
                else:
                    sig_increase = True

        if sig_decrease and not sig_increase:
            decision = "Evidence consistent with having children decreasing engagement in extramarital affairs."
        elif sig_increase and not sig_decrease:
            decision = "Evidence consistent with having children increasing engagement in extramarital affairs."
        elif sig_decrease and sig_increase:
            decision = "Mixed significant findings (some parts suggest decrease, some suggest increase)."
        else:
            decision = "No statistically significant evidence that having children changes engagement in extramarital affairs (at alpha=0.05)."
    except Exception:
        decision = "Could not form a combined decision due to errors parsing model outputs."

    description = " ; ".join(interpretation_lines) + " Overall conclusion: " + decision

    return {
        "object": {
            "logit": out['logit'],
            "zinb": out['zinb']
        },
        "description": description
    }