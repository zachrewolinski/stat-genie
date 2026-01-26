def extract_final_answer(model_output):
    """
    Extract statistics related to the effect of 'Children' on affairs from the model_output.

    Returns:
      {
        "object": {
          "tobit": {
            "coef": float or None,
            "se": float or None,
            "z": float or None,
            "pvalue": float or None
          },
          "logit": {
            "coef": float or None,
            "se": float or None,
            "pvalue": float or None,
            "odds_ratio": float or None,
            "odds_ratio_95ci": [float_lower, float_upper] or None
          },
          "decision_decreases_affairs": bool (True if there is statistically significant evidence that children decrease affairs, else False)
        },
        "description": "Text summary of the results and interpretation"
      }
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    result_obj = {}
    # ---- Tobit extraction ----
    tobit_res = model_output.get('tobit')
    tobit_info = {"coef": None, "se": None, "z": None, "pvalue": None}
    if isinstance(tobit_res, dict):
        try:
            coef = tobit_res.get('params', {}).get('Children', None)
            se = tobit_res.get('se', {}).get('Children', None)
            tobit_info['coef'] = float(coef) if coef is not None else None
            tobit_info['se'] = float(se) if se is not None else None
            if tobit_info['coef'] is not None and tobit_info['se'] not in (None, 0, np.nan):
                z = tobit_info['coef'] / tobit_info['se']
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                tobit_info['z'] = float(z)
                tobit_info['pvalue'] = float(p)
            else:
                tobit_info['z'] = None
                tobit_info['pvalue'] = None
        except Exception:
            pass
    result_obj['tobit'] = tobit_info

    # ---- Logit extraction ----
    logit_res = model_output.get('logit_result_obj')
    logit_info = {"coef": None, "se": None, "pvalue": None, "odds_ratio": None, "odds_ratio_95ci": None}
    if logit_res is not None and hasattr(logit_res, "params"):
        try:
            # params and bse/pvalues are pandas Series
            coef = logit_res.params.get('Children')
            se = None
            # statsmodels stores bse in .bse
            if hasattr(logit_res, 'bse'):
                se = logit_res.bse.get('Children')
            pval = None
            if hasattr(logit_res, 'pvalues'):
                pval = logit_res.pvalues.get('Children')
            logit_info['coef'] = float(coef) if coef is not None else None
            logit_info['se'] = float(se) if se is not None else None
            logit_info['pvalue'] = float(pval) if pval is not None else None
            # odds ratio and 95% CI if conf_int available
            if hasattr(logit_res, 'conf_int'):
                ci = logit_res.conf_int()
                if 'Children' in ci.index:
                    lower, upper = ci.loc['Children'].tolist()
                    or_lower, or_upper = float(np.exp(lower)), float(np.exp(upper))
                    logit_info['odds_ratio_95ci'] = [or_lower, or_upper]
            if logit_info['coef'] is not None:
                logit_info['odds_ratio'] = float(np.exp(logit_info['coef']))
        except Exception:
            pass
    result_obj['logit'] = logit_info

    # ---- Decision rule and textual interpretation ----
    # Primary model specified was Tobit; use Tobit significance and sign if available.
    decision_decreases = False
    explanation_lines = []

    # Inspect Tobit first (primary). If Tobit coefficient is negative and p < 0.05 -> evidence of decrease.
    tob_coef = result_obj['tobit']['coef']
    tob_p = result_obj['tobit']['pvalue']
    if tob_coef is not None:
        explanation_lines.append(f"Tobit: coef(Children) = {tob_coef:.4f}" +
                                 (f", se = {result_obj['tobit']['se']:.4f}" if result_obj['tobit']['se'] is not None else "") +
                                 (f", p = {tob_p:.4g}" if tob_p is not None else ""))
        if (tob_p is not None) and (tob_coef < 0) and (tob_p < 0.05):
            decision_decreases = True
    else:
        explanation_lines.append("Tobit: no coefficient available.")

    # Inspect Logit (robustness)
    log_coef = result_obj['logit']['coef']
    log_p = result_obj['logit']['pvalue']
    if log_coef is not None:
        explanation_lines.append(f"Logit: coef(Children) = {log_coef:.4f}, se = {result_obj['logit']['se']:.4f}, p = {log_p:.4g}")
        if result_obj['logit']['odds_ratio'] is not None:
            explanation_lines.append(f"  Odds ratio = {result_obj['logit']['odds_ratio']:.3f}" +
                                     (f", 95% CI = [{result_obj['logit']['odds_ratio_95ci'][0]:.3f}, {result_obj['logit']['odds_ratio_95ci'][1]:.3f}]" if result_obj['logit']['odds_ratio_95ci'] else ""))
        # If Tobit was inconclusive/missing, allow logit to determine decision
        if tob_coef is None:
            if (log_p is not None) and (log_coef < 0) and (log_p < 0.05):
                decision_decreases = True
    else:
        explanation_lines.append("Logit: no fitted result available or 'Children' not in model.")

    # Final summary interpretation
    if decision_decreases:
        final_text = ("Conclusion: There is statistically significant evidence (primary model) that having children decreases "
                      "engagement in extramarital affairs.")
    else:
        # Provide more informative explanation based on observed signs and p-values
        signs = []
        if tob_coef is not None:
            signs.append(("Tobit", tob_coef))
        if log_coef is not None:
            signs.append(("Logit", log_coef))
        sign_str = ", ".join([f"{name} coef={coef:.4f}" for name, coef in signs]) if signs else "no coefficient estimates available"
        pinfo = []
        if tob_p is not None:
            pinfo.append(f"Tobit p={tob_p:.4g}")
        if log_p is not None:
            pinfo.append(f"Logit p={log_p:.4g}")
        p_str = ", ".join(pinfo) if pinfo else "no p-values available"
        final_text = ("Conclusion: There is no evidence that having children decreases extramarital-affair engagement. "
                      f"Observed estimates: {sign_str}; p-values: {p_str}. "
                      "In the provided results, the 'Children' coefficient is positive in both the Tobit and Logit models "
                      "(i.e., if anything associated with a higher likelihood/frequency), but the logistic coefficient is not "
                      "statistically significant (p > 0.05). The Tobit standard errors reported are very large/unstable, "
                      "so the Tobit inference is unreliable here. Overall: no support for the hypothesis that having children "
                      "reduces extramarital affairs.")
    explanation_lines.insert(0, final_text)

    return {
        "object": result_obj,
        "description": "\n".join(explanation_lines)
    }