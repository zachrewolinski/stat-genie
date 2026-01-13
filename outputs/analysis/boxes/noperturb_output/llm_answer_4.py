def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of age (and its interaction with culture)
    from a fitted statsmodels logistic regression results object.

    Returns a dictionary with:
      - "object": dict with numeric results (coefficients, SEs, p-values, 95% CIs,
                    odds ratios and OR 95% CIs) for the age main effect and any
                    age x culture interaction terms.
      - "description": short interpretation of whether (a) there is an overall
                       age effect, and (b) whether the age effect differs across cultures
                       (based on significance of interaction terms).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Attempt to access params, bse, pvalues, conf_int
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        ci_array = res.conf_int(alpha=0.05)
    except Exception as e:
        raise ValueError("Provided model_output does not look like a statsmodels results object.") from e

    # Make CI into DataFrame with index matching params
    try:
        ci_df = pd.DataFrame(ci_array, index=params.index, columns=['2.5%', '97.5%'])
    except Exception:
        # fallback: construct with positional index
        ci_df = pd.DataFrame(ci_array, columns=['2.5%', '97.5%'])
        ci_df.index = params.index

    # Function to build term summary
    def summarize_term(name):
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else float('nan')
        p = float(pvalues[name]) if name in pvalues.index else float('nan')
        ci_lower = float(ci_df.loc[name, '2.5%'])
        ci_upper = float(ci_df.loc[name, '97.5%'])
        or_val = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))
        return {
            'term': name,
            'coef': coef,
            'se': se,
            'p_value': p,
            'ci_95': [ci_lower, ci_upper],
            'odds_ratio': or_val,
            'odds_ratio_ci_95': [or_ci_lower, or_ci_upper]
        }

    results = {}
    # Extract main age effect if present
    if 'age_centered' in params.index:
        results['age_centered'] = summarize_term('age_centered')
    else:
        # try alternate naming
        age_terms = [n for n in params.index if 'age_centered' in n and ':' not in n]
        if len(age_terms) == 1:
            results['age_centered'] = summarize_term(age_terms[0])

    # Extract interactions: any parameter name that contains age_centered and a colon
    interaction_terms = [n for n in params.index if ('age_centered' in n) and (':' in n)]
    interactions_summary = {}
    for name in interaction_terms:
        interactions_summary[name] = summarize_term(name)
    if interactions_summary:
        results['age_x_culture_interactions'] = interactions_summary

    # Also extract culture main effects if present (useful for interpretation)
    culture_terms = [n for n in params.index if (n.startswith('C(culture)'))]
    culture_summary = {}
    for name in culture_terms:
        culture_summary[name] = summarize_term(name)
    if culture_summary:
        results['culture_main_effects'] = culture_summary

    # Determine significance conclusions
    alpha = 0.05
    conclusion = {}
    # Overall age effect
    if 'age_centered' in results:
        p_age = results['age_centered']['p_value']
        if not np.isnan(p_age):
            conclusion['overall_age_effect_significant'] = (p_age < alpha)
            conclusion['overall_age_effect_p'] = p_age
        else:
            conclusion['overall_age_effect_significant'] = None
            conclusion['overall_age_effect_p'] = None
    else:
        conclusion['overall_age_effect_significant'] = None
        conclusion['overall_age_effect_p'] = None

    # Interactions: if any interaction term has p < alpha, then age effect differs by culture
    if interactions_summary:
        sig_interactions = {n: interactions_summary[n]['p_value'] for n in interactions_summary}
        sig_flags = {n: (p is not None and (not np.isnan(p)) and p < alpha) for n, p in sig_interactions.items()}
        conclusion['age_x_culture_any_significant'] = any(sig_flags.values())
        conclusion['age_x_culture_significant_terms'] = {n: {'p_value': sig_interactions[n], 'significant': sig_flags[n]}
                                                         for n in sig_interactions}
    else:
        conclusion['age_x_culture_any_significant'] = False
        conclusion['age_x_culture_significant_terms'] = {}

    # Build human-readable short description
    desc_lines = []
    if conclusion['overall_age_effect_significant'] is True:
        desc_lines.append(
            "There is a statistically significant overall effect of age on choosing the majority-demonstrated option "
            f"(age_centered coef = {results['age_centered']['coef']:.3f}, p = {results['age_centered']['p_value']:.3g}; "
            f"OR = {results['age_centered']['odds_ratio']:.3f}, 95% CI OR = [{results['age_centered']['odds_ratio_ci_95'][0]:.3f}, {results['age_centered']['odds_ratio_ci_95'][1]:.3f}])."
        )
    elif conclusion['overall_age_effect_significant'] is False:
        desc_lines.append(
            "No statistically significant overall effect of age was detected "
            f"(age_centered coef = {results['age_centered']['coef']:.3f}, p = {results['age_centered']['p_value']:.3g})."
        )
    else:
        desc_lines.append("No main age_centered term found in the model results.")

    if conclusion['age_x_culture_any_significant']:
        sig_terms = [n for n, v in conclusion['age_x_culture_significant_terms'].items() if v['significant']]
        desc_lines.append(
            "At least one age × culture interaction is statistically significant, indicating that the effect of age on majority choice "
            "differs across some cultural contexts. Significant interaction terms: " + ", ".join(sig_terms) + "."
        )
    else:
        if interaction_terms:
            desc_lines.append(
                "No age × culture interaction reached conventional significance (p < 0.05), suggesting the age effect is "
                "not detectably different across cultures in this model."
            )
        else:
            desc_lines.append("No age × culture interaction terms are present in the model results.")

    # Final return object
    return {
        "object": {
            "numeric_results": results,
            "conclusion_flags": conclusion
        },
        "description": " ".join(desc_lines)
    }