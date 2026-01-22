def extract_final_answer(model_output):
    """
    Extracts the effect of having children on number of extramarital affairs
    from the provided model_output dict containing keys 'zinb' and 'nb'.
    
    Returns a dictionary with:
      - "object": a dict with extracted statistics for each model (ZINB and NB).
      - "description": a brief human-readable interpretation answering whether
                       having children decreases engagement in extramarital affairs.
    """
    import math

    def normal_two_sided_pvalue(z):
        # two-sided p-value for Z using error function (no external libs)
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        p = 2 * (1 - cdf) if z >= 0 else 2 * cdf
        return p

    def find_param_name(params_index, base_name):
        # find param name that matches base_name but is not an inflation param
        # prefer exact match; otherwise find one that endswith base_name and not startswith 'inflate'
        if base_name in params_index:
            return base_name
        for n in params_index:
            if n.endswith(base_name) and not n.startswith('inflate'):
                return n
        # fallback: return any param that endswith base_name
        for n in params_index:
            if n.endswith(base_name):
                return n
        return None

    def extract_from_results(res, model_type='zinb'):
        """
        Extract coefficient, SE, p-value, CI, IRR for:
          - children_binary (effect for reference gender, i.e. gender_male=0)
          - children_gender_interaction (the added effect for male)
        For combined male effect we compute coef_children + coef_interaction and appropriate SE using covariance.
        """
        out = {'model_type': model_type, 'available': False}
        if res is None:
            return out

        # For ZINB, params contains both count and inflate params; use the count-part names
        params = res.params
        bse = getattr(res, 'bse', None)
        pvalues = getattr(res, 'pvalues', None)
        conf = None
        try:
            conf = res.conf_int()
        except Exception:
            conf = None

        # covariance matrix for linear combination
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

        # Find param names
        children_name = find_param_name(params.index, 'children_binary')
        inter_name = find_param_name(params.index, 'children_gender_interaction')

        if children_name is None:
            out['error'] = "Could not find parameter named like 'children_binary' in model params."
            return out

        # Extract base (female) effect
        coef_children = float(params[children_name])
        se_children = float(bse[children_name]) if bse is not None and children_name in bse.index else None
        p_children = float(pvalues[children_name]) if pvalues is not None and children_name in pvalues.index else None
        if conf is not None and children_name in conf.index:
            ci_low_children, ci_upp_children = float(conf.loc[children_name, 0]), float(conf.loc[children_name, 1])
        else:
            ci_low_children = ci_upp_children = None

        # Interaction (may be missing)
        if inter_name is not None:
            coef_inter = float(params[inter_name])
            se_inter = float(bse[inter_name]) if bse is not None and inter_name in bse.index else None
            p_inter = float(pvalues[inter_name]) if pvalues is not None and inter_name in pvalues.index else None
            if conf is not None and inter_name in conf.index:
                ci_low_inter, ci_upp_inter = float(conf.loc[inter_name, 0]), float(conf.loc[inter_name, 1])
            else:
                ci_low_inter = ci_upp_inter = None
        else:
            coef_inter = 0.0
            se_inter = None
            p_inter = None
            ci_low_inter = ci_upp_inter = None

        # Combined effect for males = coef_children + coef_inter
        coef_male = coef_children + coef_inter
        se_male = None
        p_male = None
        ci_low_male = ci_upp_male = None
        if cov is not None and children_name in cov.index:
            # Determine covariance between children and interaction if interaction exists in cov
            if inter_name is not None and inter_name in cov.index:
                var = cov.loc[children_name, children_name] + cov.loc[inter_name, inter_name] + 2 * cov.loc[children_name, inter_name]
                se_male = float(math.sqrt(var)) if var >= 0 else None
            else:
                # no interaction in cov (unlikely) -> just children se
                se_male = float(math.sqrt(cov.loc[children_name, children_name]))
        else:
            # fallback if no cov matrix, try to combine SEs (conservative)
            if se_children is not None and se_inter is not None:
                se_male = float(math.sqrt(se_children**2 + se_inter**2))
        if se_male is not None:
            z = coef_male / se_male if se_male != 0 else 0.0
            p_male = normal_two_sided_pvalue(abs(z))
            # 95% CI using normal approx
            ci_low_male = coef_male - 1.96 * se_male
            ci_upp_male = coef_male + 1.96 * se_male

        # For female (reference gender, gender_male=0)
        irr_female = math.exp(coef_children)
        irr_female_ci = (math.exp(ci_low_children) if ci_low_children is not None else None,
                         math.exp(ci_upp_children) if ci_upp_children is not None else None)

        # For male
        irr_male = math.exp(coef_male)
        irr_male_ci = (math.exp(ci_low_male) if ci_low_male is not None else None,
                       math.exp(ci_upp_male) if ci_upp_male is not None else None)

        # Aggregate outputs
        out.update({
            'available': True,
            'children_param_name': children_name,
            'children_coef': coef_children,
            'children_se': se_children,
            'children_pvalue': p_children,
            'children_ci95': (ci_low_children, ci_upp_children),
            'children_IRR': irr_female,
            'children_IRR95': irr_female_ci,
            'interaction_param_name': inter_name,
            'interaction_coef': coef_inter,
            'interaction_se': se_inter,
            'interaction_pvalue': p_inter,
            'interaction_ci95': (ci_low_inter, ci_upp_inter),
            'male_combined_coef': coef_male,
            'male_combined_se': se_male,
            'male_combined_pvalue': p_male,
            'male_combined_ci95': (ci_low_male, ci_upp_male),
            'male_IRR': irr_male,
            'male_IRR95': irr_male_ci
        })
        return out

    # Prepare output container
    results_summary = {}

    zinb_res = model_output.get('zinb')
    nb_res = model_output.get('nb')

    results_summary['zinb'] = extract_from_results(zinb_res, model_type='zinb')
    results_summary['nb'] = extract_from_results(nb_res, model_type='nb')

    # Build a concise description / final verdict
    lines = []
    def fmt_effect(x):
        if not x.get('available', False):
            return "model not available / parameter missing"
        s = []
        # Female
        s.append("Females (gender_male=0): IRR={:.3f}".format(x['children_IRR']))
        if x['children_IRR95'][0] is not None:
            s[-1] += " (95% CI [{:.3f}, {:.3f}])".format(x['children_IRR95'][0], x['children_IRR95'][1])
        if x['children_pvalue'] is not None:
            s[-1] += ", p={:.3g}".format(x['children_pvalue'])
        # Male
        s.append("Males (gender_male=1): IRR={:.3f}".format(x['male_IRR']))
        if x['male_IRR95'][0] is not None:
            s[-1] += " (95% CI [{:.3f}, {:.3f}])".format(x['male_IRR95'][0], x['male_IRR95'][1])
        if x['male_combined_pvalue'] is not None:
            s[-1] += ", p={:.3g}".format(x['male_combined_pvalue'])
        return " ; ".join(s)

    # Add per-model lines
    if results_summary['zinb'].get('available'):
        lines.append("ZINB results: " + fmt_effect(results_summary['zinb']))
    else:
        lines.append("ZINB results: not available or parameter not found.")

    if results_summary['nb'].get('available'):
        lines.append("Negative Binomial GLM results: " + fmt_effect(results_summary['nb']))
    else:
        lines.append("Negative Binomial GLM results: not available or parameter not found.")

    # Simple overall interpretation rule:
    # If both models show IRR < 1 and p < 0.05 for that sex -> strong evidence for decrease.
    # If one model shows significant IRR <1 -> some evidence. Otherwise no evidence.
    def interpret_overall(x):
        if not x.get('available'):
            return None
        res = {}
        female_sig_decrease = (x['children_pvalue'] is not None and x['children_pvalue'] < 0.05 and x['children_IRR'] < 1)
        male_sig_decrease = (x['male_combined_pvalue'] is not None and x['male_combined_pvalue'] < 0.05 and x['male_IRR'] < 1)
        res['female_sig_decrease'] = female_sig_decrease
        res['male_sig_decrease'] = male_sig_decrease
        return res

    zinb_interp = interpret_overall(results_summary['zinb'])
    nb_interp = interpret_overall(results_summary['nb'])

    # summarize across models
    female_votes = sum(1 for interp in (zinb_interp, nb_interp) if interp and interp['female_sig_decrease'])
    male_votes = sum(1 for interp in (zinb_interp, nb_interp) if interp and interp['male_sig_decrease'])

    overall_lines = []
    if female_votes == 2:
        overall_lines.append("Strong evidence across both models that having children decreases affairs for females.")
    elif female_votes == 1:
        overall_lines.append("Some evidence (one model) that having children decreases affairs for females.")
    else:
        overall_lines.append("No consistent evidence that having children decreases affairs for females.")

    if male_votes == 2:
        overall_lines.append("Strong evidence across both models that having children decreases affairs for males.")
    elif male_votes == 1:
        overall_lines.append("Some evidence (one model) that having children decreases affairs for males.")
    else:
        overall_lines.append("No consistent evidence that having children decreases affairs for males.")

    description = " | ".join(lines) + " || Overall: " + " ".join(overall_lines)

    return {
        "object": results_summary,
        "description": description
    }