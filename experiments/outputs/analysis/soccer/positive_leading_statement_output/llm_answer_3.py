def extract_final_answer(model_output):
    """
    Extracts the DarkSkin coefficient and related statistics from the provided
    model_output dictionary (expected keys: 'neg_binom_result', 'poisson_result',
    'n_obs', 'n_referees').

    Returns a dictionary with:
      - "object": a dict containing numeric results for each model (coef, se, z,
                  p-value, 95% CI, incidence-rate-ratio and its 95% CI),
                  plus sample sizes.
      - "description": a concise interpretation of the results answering whether
                       darker-skinned players are more likely to receive red cards.
    """
    import numpy as np

    out = {
        'n_obs': None,
        'n_referees': None,
        'models': {}
    }

    try:
        out['n_obs'] = int(model_output.get('n_obs', None))
        out['n_referees'] = int(model_output.get('n_referees', None))
    except Exception:
        out['n_obs'] = model_output.get('n_obs', None)
        out['n_referees'] = model_output.get('n_referees', None)

    model_names = {
        'neg_binom_result': 'NegativeBinomial',
        'poisson_result': 'Poisson'
    }

    for key, pretty in model_names.items():
        res = model_output.get(key, None)
        if res is None:
            out['models'][pretty] = {'error': f'{key} not found in model_output'}
            continue

        try:
            # coefficient, robust se, p-value, conf int
            coef = float(res.params['DarkSkin'])
            se = float(res.bse['DarkSkin'])
            pval = float(res.pvalues['DarkSkin'])
            ci = res.conf_int().loc['DarkSkin'].values  # [lower, upper]
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
            z = coef / se if se != 0 else None

            irr = float(np.exp(coef))
            irr_ci_lower = float(np.exp(ci_lower))
            irr_ci_upper = float(np.exp(ci_upper))

            # Round for readability
            def r(x): return None if x is None else float(np.round(x, 4))

            out['models'][pretty] = {
                'coef': r(coef),
                'se': r(se),
                'z': r(z),
                'p_value': r(pval),
                'ci_lower': r(ci_lower),
                'ci_upper': r(ci_upper),
                'incidence_rate_ratio': r(irr),
                'irr_ci_lower': r(irr_ci_lower),
                'irr_ci_upper': r(irr_ci_upper),
                # significance flag
                'significant_0.05': (pval < 0.05)
            }
        except Exception as e:
            out['models'][pretty] = {'error': f'Failed to extract stats: {str(e)}'}

    # Interpret results across models
    interpretations = []
    try:
        nb = out['models'].get('NegativeBinomial', {})
        pois = out['models'].get('Poisson', {})

        def interp(model_dict, name):
            if 'error' in model_dict:
                return f"{name}: {model_dict.get('error')}"
            sign = model_dict['significant_0.05']
            irr = model_dict['incidence_rate_ratio']
            ci_low = model_dict['irr_ci_lower']
            ci_up = model_dict['irr_ci_upper']
            direction = 'higher' if irr > 1 else 'lower' if irr < 1 else 'no difference'
            sig_text = 'statistically significant (p < 0.05)' if sign else 'not statistically significant (p >= 0.05)'
            return (f"{name}: IRR = {irr} (95% CI: {ci_low}–{ci_up}), meaning the red-card rate for "
                    f"dark-skinned players is {direction} compared to light-skinned players; {sig_text}.")

        if nb:
            interpretations.append(interp(nb, 'Negative Binomial'))
        if pois:
            interpretations.append(interp(pois, 'Poisson'))

        # Overall conclusion logic
        concl = "Overall conclusion: "
        nb_sig = isinstance(nb, dict) and ('significant_0.05' in nb and nb['significant_0.05'])
        pois_sig = isinstance(pois, dict) and ('significant_0.05' in pois and pois['significant_0.05'])
        # Determine direction if significant
        nb_dir = None
        pois_dir = None
        if nb and 'incidence_rate_ratio' in nb:
            nb_dir = 'higher' if nb['incidence_rate_ratio'] > 1 else 'lower' if nb['incidence_rate_ratio'] < 1 else 'no difference'
        if pois and 'incidence_rate_ratio' in pois:
            pois_dir = 'higher' if pois['incidence_rate_ratio'] > 1 else 'lower' if pois['incidence_rate_ratio'] < 1 else 'no difference'

        if nb_sig and pois_sig and nb_dir == pois_dir:
            concl += f"Yes — both models find a {nb_dir} red-card rate for dark-skinned players (results are statistically significant)."
        elif (nb_sig and not pois_sig) or (pois_sig and not nb_sig):
            which = 'Negative Binomial' if nb_sig and not pois_sig else 'Poisson'
            dirn = nb_dir if nb_sig else pois_dir
            concl += (f"Inconclusive: {which} model shows a statistically significant {dirn} rate for dark-skinned players, "
                     f"but the other model does not reach significance. Interpretation should be cautious.")
        elif not nb_sig and not pois_sig:
            concl += "No evidence that dark-skinned players receive more red cards — neither model shows a statistically significant effect."
        else:
            concl += "Mixed or unclear evidence."

    except Exception as e:
        concl = f"Could not form an overall interpretation: {str(e)}"

    description = " ; ".join(interpretations) + " || " + concl

    return {
        "object": out,
        "description": description
    }