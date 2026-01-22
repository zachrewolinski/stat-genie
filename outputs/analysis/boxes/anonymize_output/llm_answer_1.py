def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for age effects
    and Age x Site interactions from:
      - a fitted statsmodels MNLogit result (model_output['mnlogit_result'])
      - a fitted statsmodels Logit result (model_output['logit_result'])
    Returns a dict with keys:
      - "object": dict with numeric summaries for multinomial (per outcome) and binary models
      - "description": brief interpretation focused on (1) Age main effect and (2) Age x Site interactions
    """
    import numpy as np

    out = {"multinomial": None, "binary": None}
    desc_lines = []

    # Helper to build term summary (coef, se, p, ci)
    def summarize_terms(params, bse, pvals, terms, label_for_outcome=None):
        summ = {}
        for t in terms:
            if t not in params.index:
                continue
            coef = float(params.loc[t])
            se = float(bse.loc[t]) if (t in bse.index) else np.nan
            p = float(pvals.loc[t]) if (t in pvals.index) else np.nan
            ci_low = coef - 1.96 * se if not np.isnan(se) else np.nan
            ci_high = coef + 1.96 * se if not np.isnan(se) else np.nan
            summ[t] = {
                "coef": coef,
                "se": se,
                "pvalue": p,
                "ci95_low": ci_low,
                "ci95_high": ci_high
            }
        return summ

    # --- Multinomial results extraction ---
    mn = model_output.get('mnlogit_result')
    if isinstance(mn, dict) and 'error' in mn:
        out['multinomial'] = {"error": mn['error']}
        desc_lines.append("Multinomial model failed: " + mn['error'])
    else:
        try:
            params_df = mn.params  # DataFrame: rows = outcomes (e.g., 1,2), cols = exog names
            bse_df = mn.bse
            pvals_df = mn.pvalues

            multinom_summary = {}
            # Determine relevant term names: AgeYears and Age_x_* interaction terms
            # Columns of params_df are the exog names
            exog_terms = list(params_df.columns)
            age_term = "AgeYears"
            age_x_terms = [c for c in exog_terms if c.startswith("Age_x_")]

            # For each outcome (e.g., 1=majority, 2=minority), extract summaries
            for outcome in params_df.index:
                row_params = params_df.loc[outcome]
                row_bse = bse_df.loc[outcome]
                row_p = pvals_df.loc[outcome]
                terms_to_get = [age_term] + age_x_terms
                summ = summarize_terms(row_params, row_bse, row_p, terms_to_get, label_for_outcome=outcome)
                multinom_summary[str(outcome)] = summ

                # Interpret for the 'majority' outcome (ChoiceNum==1) if present
                # We will add one sentence of interpretation after loop.
            out['multinomial'] = multinom_summary

            # Simple interpretation logic for majority (outcome index '1' expected)
            maj_key = None
            # params_df.index might be integers; find the one corresponding to '1'
            for idx in params_df.index:
                if str(idx) == '1' or idx == 1:
                    maj_key = idx
                    break
            if maj_key is not None:
                maj_params = params_df.loc[maj_key]
                maj_p = pvals_df.loc[maj_key]
                age_p = float(maj_p.get(age_term, np.nan)) if age_term in maj_p.index else np.nan
                sig_age = (not np.isnan(age_p)) and (age_p < 0.05)
                # any significant interactions?
                sig_interactions = {}
                for t in age_x_terms:
                    p = float(maj_p.get(t, np.nan)) if t in maj_p.index else np.nan
                    if (not np.isnan(p)) and (p < 0.05):
                        sig_interactions[t] = p
                if sig_age and (len(sig_interactions) == 0):
                    desc_lines.append("Multinomial (majority vs unchosen): AgeYears shows a significant main effect (p < 0.05) with no significant Age x Site interactions -> consistent developmental change across sites.")
                elif (not sig_age) and (len(sig_interactions) > 0):
                    desc_lines.append(f"Multinomial (majority vs unchosen): No significant overall AgeYears main effect, but significant Age x Site interactions found for terms: {list(sig_interactions.keys())} -> developmental trajectories differ across sites.")
                elif sig_age and (len(sig_interactions) > 0):
                    desc_lines.append(f"Multinomial (majority vs unchosen): Significant AgeYears main effect and some significant Age x Site interactions ({list(sig_interactions.keys())}) -> there is an overall developmental trend but its magnitude/direction varies across sites.")
                else:
                    desc_lines.append("Multinomial (majority vs unchosen): No evidence of a significant AgeYears main effect or Age x Site interactions (no p < 0.05) -> no developmental change in majority choice detected across or between sites.")
            else:
                desc_lines.append("Multinomial: Could not locate outcome corresponding to 'majority' (1); full parameter table is returned for inspection.")
        except Exception as e:
            out['multinomial'] = {"error": f"Exception extracting multinomial results: {e}"}
            desc_lines.append("Error extracting multinomial results: " + str(e))

    # --- Binary (DemonstratedChosen) logistic results extraction ---
    log = model_output.get('logit_result')
    if isinstance(log, dict) and 'error' in log:
        out['binary'] = {"error": log['error']}
        desc_lines.append("Binary logistic model failed: " + log['error'])
    else:
        try:
            params = log.params  # Series
            bse = log.bse
            pvals = log.pvalues

            # Terms of interest: AgeYears and any AgeYears:C(SiteID) interaction terms.
            terms = []
            if 'AgeYears' in params.index:
                terms.append('AgeYears')
            # Interaction terms usually contain 'AgeYears' in their name
            interaction_terms = [t for t in params.index if 'AgeYears' in t and t != 'AgeYears']
            terms += interaction_terms

            binary_summary = summarize_terms(params, bse, pvals, terms)
            out['binary'] = binary_summary

            # Interpret
            age_p = float(pvals.get('AgeYears', np.nan)) if 'AgeYears' in pvals.index else np.nan
            sig_age = (not np.isnan(age_p)) and (age_p < 0.05)
            sig_interactions = {t: float(pvals[t]) for t in interaction_terms if (t in pvals.index and (not np.isnan(pvals[t])) and pvals[t] < 0.05)}
            if sig_age and (len(sig_interactions) == 0):
                desc_lines.append("Binary (any demonstrated option chosen): AgeYears has a significant main effect (p < 0.05) with no significant Age x Site interactions -> general reliance on social information changes with age similarly across sites.")
            elif (not sig_age) and (len(sig_interactions) > 0):
                desc_lines.append(f"Binary: No overall AgeYears main effect, but significant Age x Site interactions for: {list(sig_interactions.keys())} -> reliance on social information changes with age in a site-dependent way.")
            elif sig_age and (len(sig_interactions) > 0):
                desc_lines.append(f"Binary: Significant AgeYears main effect and some significant Age x Site interactions ({list(sig_interactions.keys())}) -> overall age-related change in social reliance, with site-specific modulation.")
            else:
                desc_lines.append("Binary: No evidence that general reliance on demonstrated options changes with age across or between sites (no p < 0.05).")
        except Exception as e:
            out['binary'] = {"error": f"Exception extracting binary-logit results: {e}"}
            desc_lines.append("Error extracting binary results: " + str(e))

    description = " ".join(desc_lines)
    return {"object": out, "description": description}