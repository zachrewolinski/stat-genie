def extract_final_answer(model_output):
    """
    Extracts the estimated effect of the name-femininity independent variable from a model output.
    Returns a dictionary with keys:
      - "object": dict with keys: 'variable', 'coef', 'p_value' (or None), 'conf_int' (or None),
                          'r_squared' (or None), 'n_obs' (or None)
      - "description": human-readable interpretation in the study context
    The function handles:
      - statsmodels regression result objects
      - a dict of statsmodels results
      - a plain dict that contains a 'coefficients' mapping (as in the provided example)
    """
    import math
    import numpy as np

    # Candidates for the femininity variable (various possible names used in code/output)
    continuous_names = ['MasFeminine_z', 'MasFeminine', 'masfem', 'MasFeminineMTurk_z', 'masfem_mturk', 'MasFeminine_z_std']
    binary_names = ['FemaleName', 'female', 'gender_mf', 'GenderMF', 'is_female']
    all_candidates = continuous_names + binary_names

    # Helper to build return object
    def build_object(varname, coef, pval=None, ci=None, r2=None, nobs=None, var_type='continuous'):
        obj = {
            'variable': varname,
            'coef': None if coef is None else float(coef),
            'p_value': None if pval is None else (float(pval) if not np.isnan(pval) else None),
            'conf_int': None if ci is None else [float(ci[0]), float(ci[1])],
            'r_squared': None if r2 is None else float(r2),
            'n_obs': None if nobs is None else int(nobs),
            'variable_type': var_type  # 'continuous' or 'binary'
        }

        # Build description text
        desc_lines = []
        if var_type == 'continuous':
            desc_lines.append(
                "Coefficient interprets as the change in Log(Deaths + 1) per 1 SD increase in name femininity."
            )
        else:
            desc_lines.append(
                "Coefficient interprets as the difference in Log(Deaths + 1) for hurricanes with female names (1) vs male names (0)."
            )

        if coef is None:
            desc_lines.append("No coefficient found for the femininity variable in the model output.")
        else:
            desc_lines.append(f"Estimated coefficient = {obj['coef']:.6g}.")
            # multiplicative effect on (Deaths+1)
            try:
                mult = math.exp(obj['coef'])
                desc_lines.append(f"Equivalent multiplicative effect on (Deaths + 1): exp(coef) = {mult:.6g}.")
            except Exception:
                pass

            # Practical significance remark if near zero
            if abs(obj['coef']) < 1e-6:
                desc_lines.append("The estimate is essentially zero (no meaningful effect).")
            else:
                desc_lines.append("Non-zero point estimate observed; statistical significance unknown here if p-value is not available.")

        if obj['p_value'] is not None:
            desc_lines.append(f"Reported p-value = {obj['p_value']:.4g}.")
        else:
            desc_lines.append("No p-value available in the provided output; cannot make a definitive statistical-significance claim.")

        if obj['conf_int'] is not None:
            desc_lines.append(f"95% CI = [{obj['conf_int'][0]:.6g}, {obj['conf_int'][1]:.6g}].")

        if obj['r_squared'] is not None:
            desc_lines.append(f"Model R-squared = {obj['r_squared']:.4g} (if meaningful).")
        if obj['n_obs'] is not None:
            desc_lines.append(f"Number of observations = {obj['n_obs']}.")

        full_desc = " ".join(desc_lines)
        return {'object': obj, 'description': full_desc}

    # Case 1: statsmodels result object
    try:
        # detect statsmodels-like result with .params
        if hasattr(model_output, 'params'):
            res = model_output
            # find candidate
            for cand in all_candidates:
                if cand in res.params.index:
                    var_type = 'continuous' if cand in continuous_names else 'binary'
                    coef = res.params[cand]
                    pval = res.pvalues[cand] if hasattr(res, 'pvalues') else None
                    ci = None
                    if hasattr(res, 'conf_int'):
                        try:
                            ci = list(res.conf_int().loc[cand].values)
                        except Exception:
                            ci = None
                    r2 = res.rsquared if hasattr(res, 'rsquared') else None
                    nobs = res.nobs if hasattr(res, 'nobs') else None
                    return build_object(cand, coef, pval, ci, r2, nobs, var_type)
            # no candidate found
            return build_object(None, None)
    except Exception:
        pass

    # Case 2: dict of statsmodels results or plain dict
    if isinstance(model_output, dict):
        # If dict may contain multiple models (values are statsmodels results objects)
        for k, v in model_output.items():
            if hasattr(v, 'params'):
                # recursive call to handle this statsmodels result
                return extract_final_answer(v)

        # If the dict is the simplified representation (like the example), look for 'coefficients' mapping
        coeffs = None
        if 'coefficients' in model_output and isinstance(model_output['coefficients'], dict):
            coeffs = model_output['coefficients']
        elif 'params' in model_output and isinstance(model_output['params'], dict):
            coeffs = model_output['params']

        r2 = model_output.get('r_squared', None)
        nobs = model_output.get('n_obs', model_output.get('nobs', None))

        if coeffs is not None:
            # normalize keys to compare ignoring case
            lower_to_key = {k.lower(): k for k in coeffs.keys()}
            found_key = None
            found_type = None
            for cand in all_candidates:
                if cand.lower() in lower_to_key:
                    found_key = lower_to_key[cand.lower()]
                    found_type = 'continuous' if cand in continuous_names else 'binary'
                    break
            if found_key is not None:
                coef = coeffs.get(found_key)
                # try to extract p-value/conf int if present under other fields (unlikely here)
                pval = None
                ci = None
                # Build and return
                return build_object(found_key, coef, pval, ci, r2, nobs, found_type)

        # If no coefficients dict or no candidate found, as a fallback attempt to parse flattened numeric keys
        # Try common shortened keys used in some outputs: 'masfem', 'gender_mf', 'female'
        flat_keys = []
        for k in model_output.keys():
            if isinstance(model_output[k], (int, float)) or (hasattr(model_output[k], '__float__') and not isinstance(model_output[k], str)):
                flat_keys.append(k)
        # search flat keys
        for fk in flat_keys:
            if fk.lower() in [c.lower() for c in all_candidates]:
                return build_object(fk, float(model_output[fk]), None, None, model_output.get('r_squared', None), model_output.get('n_obs', None),
                                    'continuous' if fk.lower() in [c.lower() for c in continuous_names] else 'binary')

    # If all attempts fail, return a null object with explanation
    return {
        'object': None,
        'description': (
            "Could not locate a femininity-variable coefficient in the supplied model_output. "
            "Supported inputs: statsmodels results object, dict of statsmodels results, or dict containing a "
            "'coefficients' mapping. Ensure the model output includes a parameter named one of: "
            + ", ".join(all_candidates)
        )
    }