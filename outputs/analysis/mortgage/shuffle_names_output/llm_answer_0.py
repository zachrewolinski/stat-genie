def extract_final_answer(model_output):
    """
    Extracts the estimated effect of gender (variable 'Female') on mortgage approval
    from the model output produced by the modeling function in the prompt.

    Returns a dict with keys:
      - "object": a dict containing numeric results (coef, odds_ratio, pvalue, CIs, decision)
      - "description": a short, plain-language interpretation of whether gender affects approval
    """
    import math
    import numpy as np
    import pandas as pd

    # Helper to build a consistent return object when Female is missing
    def not_found():
        return {
            "object": None,
            "description": "The model output does not contain a coefficient/index for 'Female'; cannot determine effect."
        }

    # Try summary_table first (contains OR, CIs, pvalue)
    if isinstance(model_output, dict) and 'summary_table' in model_output and model_output['summary_table'] is not None:
        st = model_output['summary_table']
        if 'Female' in st.index:
            row = st.loc['Female']
            # summary_table stores OR and pvalues and CI in odds ratio scale
            try:
                or_est = float(row.get('OR', np.nan))
            except Exception:
                or_est = float(np.nan)
            try:
                ci_low_or = float(row.get('2.5%', np.nan))
                ci_high_or = float(row.get('97.5%', np.nan))
            except Exception:
                ci_low_or, ci_high_or = (float(np.nan), float(np.nan))
            try:
                pval = float(row.get('pvalue', np.nan))
            except Exception:
                pval = float(np.nan)

            # Determine significance at alpha=0.05 (two-sided)
            significant = (not math.isnan(pval)) and (pval < 0.05)

            # Direction: compare OR to 1
            if not math.isnan(or_est):
                if or_est > 1:
                    direction = "Female applicants appear more likely to be approved (OR > 1)."
                elif or_est < 1:
                    direction = "Female applicants appear less likely to be approved (OR < 1)."
                else:
                    direction = "No estimated difference (OR = 1)."
            else:
                direction = "Direction unknown (OR unavailable)."

            # Compose interpretation text, with caution about CI/pvalues
            if math.isnan(pval):
                significance_text = "p-value unavailable; cannot assess statistical significance."
            elif significant:
                significance_text = f"The effect is statistically significant (p = {pval:.3g})."
            else:
                significance_text = f"There is no statistically significant evidence of a gender effect (p = {pval:.3g})."

            ci_text = f"Estimated OR = {or_est:.3g}; 95% CI for OR ≈ [{ci_low_or if not math.isinf(ci_low_or) else '-inf'}, {ci_high_or if not math.isinf(ci_high_or) else 'inf'}]."

            description = " ".join([direction, significance_text, ci_text,
                                    "Interpretation should be treated cautiously if confidence intervals are degenerate or extremely wide (indicative of separation or unstable estimates)."])

            return {
                "object": {
                    "variable": "Female",
                    "odds_ratio": or_est,
                    "ci_odds_ratio": [ci_low_or, ci_high_or],
                    "pvalue": pval,
                    "significant_at_0.05": bool(significant),
                    "note": "Results from summary_table (odds-ratio scale)."
                },
                "description": description
            }
        # fallthrough if Female not in summary_table

    # Otherwise try robust_result (contains params, pvalues, conf_int)
    if isinstance(model_output, dict) and 'robust_result' in model_output and model_output['robust_result'] is not None:
        rr = model_output['robust_result']
        params = getattr(rr, 'params', None)
        pvalues = getattr(rr, 'pvalues', None)
        conf_int_func = getattr(rr, 'conf_int', None)

        if isinstance(params, (dict, pd.Series)) and 'Female' in params:
            coef = float(params['Female'])
            pval = float(pvalues['Female']) if (pvalues is not None and 'Female' in pvalues) else float(np.nan)

            # conf_int() may be a function returning DataFrame
            try:
                conf_df = conf_int_func()
                # conf_df rows correspond to parameter names
                if 'Female' in conf_df.index:
                    ci_low_coef = float(conf_df.loc['Female'].iloc[0])
                    ci_high_coef = float(conf_df.loc['Female'].iloc[1])
                else:
                    # fallback by position if index missing
                    ci_low_coef = float(conf_df.iloc[params.index.get_loc('Female'), 0])
                    ci_high_coef = float(conf_df.iloc[params.index.get_loc('Female'), 1])
            except Exception:
                ci_low_coef, ci_high_coef = (float(np.nan), float(np.nan))

            # Convert to OR scale
            try:
                or_est = float(np.exp(coef))
                ci_low_or = float(np.exp(ci_low_coef)) if not math.isnan(ci_low_coef) else float(np.nan)
                ci_high_or = float(np.exp(ci_high_coef)) if not math.isnan(ci_high_coef) else float(np.nan)
            except Exception:
                or_est = float(np.nan)
                ci_low_or, ci_high_or = (float(np.nan), float(np.nan))

            significant = (not math.isnan(pval)) and (pval < 0.05)

            if or_est > 1:
                direction = "Female applicants appear more likely to be approved (OR > 1)."
            elif or_est < 1:
                direction = "Female applicants appear less likely to be approved (OR < 1)."
            else:
                direction = "No estimated difference (OR = 1)."

            if math.isnan(pval):
                significance_text = "p-value unavailable; cannot assess statistical significance."
            elif significant:
                significance_text = f"The effect is statistically significant (p = {pval:.3g})."
            else:
                significance_text = f"There is no statistically significant evidence of a gender effect (p = {pval:.3g})."

            ci_text = f"Coef = {coef:.3g}, 95% CI (coef) ≈ [{ci_low_coef if not math.isinf(ci_low_coef) else '-inf'}, {ci_high_coef if not math.isinf(ci_high_coef) else 'inf'}]; OR = {or_est:.3g}, 95% CI (OR) ≈ [{ci_low_or if not math.isinf(ci_low_or) else '-inf'}, {ci_high_or if not math.isinf(ci_high_or) else 'inf'}]."

            description = " ".join([direction, significance_text, ci_text,
                                    "Note: extremely large standard errors or degenerate CIs indicate unstable estimates or separation issues; interpret with caution."])

            return {
                "object": {
                    "variable": "Female",
                    "coef": coef,
                    "ci_coef": [ci_low_coef, ci_high_coef],
                    "odds_ratio": or_est,
                    "ci_odds_ratio": [ci_low_or, ci_high_or],
                    "pvalue": pval,
                    "significant_at_0.05": bool(significant),
                    "note": "Results from robust_result (coef scale with HC1 robust SEs)."
                },
                "description": description
            }

    # As a final attempt, inspect the raw fitted result.params if present
    if isinstance(model_output, dict) and 'result' in model_output and model_output['result'] is not None:
        res = model_output['result']
        # statsmodels result objects usually expose .params and .bse/.pvalues; attempt to read them
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            conf_int = None
            try:
                conf_int = res.conf_int()
            except Exception:
                conf_int = None

            if isinstance(params, (dict, pd.Series)) and 'Female' in params:
                coef = float(params['Female'])
                pval = float(pvalues['Female']) if (pvalues is not None and 'Female' in pvalues) else float(np.nan)

                if conf_int is not None and 'Female' in conf_int.index:
                    ci_low_coef = float(conf_int.loc['Female'].iloc[0])
                    ci_high_coef = float(conf_int.loc['Female'].iloc[1])
                else:
                    ci_low_coef, ci_high_coef = (float(np.nan), float(np.nan))

                or_est = float(np.exp(coef)) if not math.isnan(coef) else float(np.nan)
                ci_low_or = float(np.exp(ci_low_coef)) if not math.isnan(ci_low_coef) else float(np.nan)
                ci_high_or = float(np.exp(ci_high_coef)) if not math.isnan(ci_high_coef) else float(np.nan)
                significant = (not math.isnan(pval)) and (pval < 0.05)

                if or_est > 1:
                    direction = "Female applicants appear more likely to be approved (OR > 1)."
                elif or_est < 1:
                    direction = "Female applicants appear less likely to be approved (OR < 1)."
                else:
                    direction = "No estimated difference (OR = 1)."

                if math.isnan(pval):
                    significance_text = "p-value unavailable; cannot assess statistical significance."
                elif significant:
                    significance_text = f"The effect is statistically significant (p = {pval:.3g})."
                else:
                    significance_text = f"There is no statistically significant evidence of a gender effect (p = {pval:.3g})."

                ci_text = f"Coef = {coef:.3g}, 95% CI (coef) ≈ [{ci_low_coef}, {ci_high_coef}]; OR ≈ {or_est:.3g}, 95% CI (OR) ≈ [{ci_low_or}, {ci_high_or}]."

                description = " ".join([direction, significance_text, ci_text,
                                        "Interpret cautiously if standard errors or CIs are extremely large or degenerate."])

                return {
                    "object": {
                        "variable": "Female",
                        "coef": coef,
                        "ci_coef": [ci_low_coef, ci_high_coef],
                        "odds_ratio": or_est,
                        "ci_odds_ratio": [ci_low_or, ci_high_or],
                        "pvalue": pval,
                        "significant_at_0.05": bool(significant),
                        "note": "Results from raw result object."
                    },
                    "description": description
                }
        except Exception:
            pass

    # If nothing found
    return not_found()