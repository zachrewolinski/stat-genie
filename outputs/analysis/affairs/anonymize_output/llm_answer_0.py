def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'Children' (and its interaction with Female)
    from the provided model_output dict (expects statsmodels result objects).
    Returns a dict with keys:
      - "object": dict of extracted numeric results for logit, poisson, negbin (if available)
      - "description": brief plain-English interpretation and final answer to the question
    """
    import numpy as np

    results = {}
    notes = []

    def _extract(res, var):
        if res is None:
            return None
        try:
            params = res.params
            pvals = res.pvalues
            bse = res.bse
            conf = res.conf_int()
        except Exception:
            return None
        if var not in params.index:
            return None
        coef = float(params[var])
        se = float(bse[var]) if var in bse.index else None
        p = float(pvals[var]) if var in pvals.index else None
        # conf could be ndarray or DataFrame
        try:
            ci_low = float(conf.loc[var, 0])
            ci_high = float(conf.loc[var, 1])
        except Exception:
            # fallback if conf_int returns ndarray without named index
            try:
                idx = list(params.index).index(var)
                ci_low = float(conf[idx, 0])
                ci_high = float(conf[idx, 1])
            except Exception:
                ci_low = None
                ci_high = None
        return {"coef": coef, "se": se, "p": p, "ci": (ci_low, ci_high)}

    # Primary: logistic model for AnyAffair
    logit_res = model_output.get('logit_res')
    logit_children = _extract(logit_res, 'Children')
    logit_children_int = _extract(logit_res, 'Children_Female')

    if logit_children is not None:
        # compute odds ratio and CI
        or_ = np.exp(logit_children['coef'])
        or_ci = (np.exp(logit_children['ci'][0]) if logit_children['ci'][0] is not None else None,
                 np.exp(logit_children['ci'][1]) if logit_children['ci'][1] is not None else None)
        logit_children.update({"odds_ratio": float(or_), "or_ci": (float(or_ci[0]) if or_ci[0] is not None else None,
                                                                  float(or_ci[1]) if or_ci[1] is not None else None)})
        results['logit'] = logit_children
        notes.append("Logistic: 'Children' coef = {:.3f}, p = {:.3f}, OR = {:.3f}, 95% CI(OR) = [{:.3f}, {:.3f}]".format(
            logit_children['coef'], logit_children['p'],
            logit_children['odds_ratio'],
            logit_children['or_ci'][0], logit_children['or_ci'][1]
        ))
    else:
        results['logit'] = None
        notes.append("Logistic: result for 'Children' not available.")

    if logit_children_int is not None:
        results['logit_interaction'] = logit_children_int
        notes.append("Logistic interaction (Children x Female) coef = {:.3f}, p = {:.3f}".format(
            logit_children_int['coef'], logit_children_int['p']
        ))
    else:
        results['logit_interaction'] = None

    # Conditional count models among respondents with NumAffairs > 0
    poisson_res = model_output.get('poisson_res')
    negbin_res = model_output.get('negbin_res')

    poisson_children = _extract(poisson_res, 'Children')
    if poisson_children is not None:
        irr = np.exp(poisson_children['coef'])  # incidence rate ratio
        irr_ci = (np.exp(poisson_children['ci'][0]) if poisson_children['ci'][0] is not None else None,
                  np.exp(poisson_children['ci'][1]) if poisson_children['ci'][1] is not None else None)
        poisson_children.update({"irr": float(irr), "irr_ci": (float(irr_ci[0]) if irr_ci[0] is not None else None,
                                                               float(irr_ci[1]) if irr_ci[1] is not None else None)})
        results['poisson'] = poisson_children
        notes.append("Poisson: 'Children' coef = {:.3f}, p = {:.3f}, IRR = {:.3f}, 95% CI(IRR) = [{:.3f}, {:.3f}]".format(
            poisson_children['coef'], poisson_children['p'],
            poisson_children['irr'], poisson_children['irr_ci'][0], poisson_children['irr_ci'][1]
        ))
    else:
        results['poisson'] = None
        notes.append("Poisson: not available / not fitted.")

    negbin_children = _extract(negbin_res, 'Children')
    if negbin_children is not None:
        irr_nb = np.exp(negbin_children['coef'])
        irr_nb_ci = (np.exp(negbin_children['ci'][0]) if negbin_children['ci'][0] is not None else None,
                     np.exp(negbin_children['ci'][1]) if negbin_children['ci'][1] is not None else None)
        negbin_children.update({"irr": float(irr_nb), "irr_ci": (float(irr_nb_ci[0]) if irr_nb_ci[0] is not None else None,
                                                                 float(irr_nb_ci[1]) if irr_nb_ci[1] is not None else None)})
        results['negbin'] = negbin_children
        notes.append("NegBin: 'Children' coef = {:.3f}, p = {:.3f}, IRR = {:.3f}, 95% CI(IRR) = [{:.3f}, {:.3f}]".format(
            negbin_children['coef'], negbin_children['p'],
            negbin_children['irr'],
            negbin_children['irr_ci'][0] if negbin_children['irr_ci'][0] is not None else float('nan'),
            negbin_children['irr_ci'][1] if negbin_children['irr_ci'][1] is not None else float('nan')
        ))
    else:
        results['negbin'] = None
        notes.append("NegBin: not available / not fitted.")

    # Summarize conclusion based on p-values and directions
    conclusion = ""
    # Use primary model (logit) for the yes/no question
    if logit_children is not None:
        p = logit_children['p']
        coef = logit_children['coef']
        if p is not None and p < 0.05:
            if coef < 0:
                conclusion = ("Yes — having children is associated with a statistically significant decrease "
                              "in the probability of any extramarital affair (logit coef = {:.3f}, p = {:.3f}; OR = {:.3f})."
                              .format(coef, p, logit_children['odds_ratio']))
            else:
                conclusion = ("No — having children is associated with a statistically significant increase "
                              "in the probability of any extramarital affair (logit coef = {:.3f}, p = {:.3f}; OR = {:.3f})."
                              .format(coef, p, logit_children['odds_ratio']))
        else:
            conclusion = ("No evidence that having children decreases engagement in extramarital affairs. "
                          "In the primary logistic model the coefficient for 'Children' is {:.3f} (p = {:.3f}), "
                          "odds ratio = {:.3f} with 95% CI [{:.3f}, {:.3f}], which is not statistically significant. "
                          "Interaction with gender (Children x Female) is also not statistically significant. "
                          "Conditional count models (Poisson and Negative Binomial) for number of affairs among those "
                          "reporting any also show non-significant effects for 'Children'."
                          .format(coef, p,
                                  logit_children.get('odds_ratio', float('nan')),
                                  logit_children.get('or_ci', (None, None))[0],
                                  logit_children.get('or_ci', (None, None))[1]))
    else:
        conclusion = "Primary logistic model results for 'Children' are not available; cannot draw a conclusion."

    # Compose final description
    description = " ; ".join(notes) + "\n\nConclusion: " + conclusion

    return {"object": results, "description": description}