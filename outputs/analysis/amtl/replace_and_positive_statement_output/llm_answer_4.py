def extract_final_answer(model_output):
    """
    Extracts statistics related to the 'is_human' effect from the model_output
    produced by the provided modeling function.

    Returns a dict with:
      - "object": dictionary of extracted numeric results (coef, OR, CI, p_value, note, conclusion)
      - "description": short textual interpretation answering whether modern humans
                       have higher AMTL after accounting for controls.
    """
    import math
    import numpy as np

    result_obj = {
        'coef': None,
        'OR': None,
        'CI_lower': None,
        'CI_upper': None,
        'p_value': None,
        'note': None,
        'conclusion': None
    }

    # Helpers
    def is_finite(x):
        try:
            return np.isfinite(x)
        except Exception:
            return False

    # Try to get odds-ratio table row for 'is_human'
    try:
        or_table = model_output.get('odds_ratio_table', None) if isinstance(model_output, dict) else None
        if or_table is not None and 'is_human' in getattr(or_table, 'index', []):
            row = or_table.loc['is_human']
            # Some values might be pandas scalars; convert to float where possible
            try:
                result_obj['coef'] = float(row.get('coef', row['coef']))
            except Exception:
                result_obj['coef'] = row.get('coef', None)
            try:
                result_obj['OR'] = float(row.get('OR', row['OR']))
            except Exception:
                result_obj['OR'] = row.get('OR', None)
            try:
                result_obj['CI_lower'] = float(row.get('CI_lower', row['CI_lower']))
            except Exception:
                result_obj['CI_lower'] = row.get('CI_lower', None)
            try:
                result_obj['CI_upper'] = float(row.get('CI_upper', row['CI_upper']))
            except Exception:
                result_obj['CI_upper'] = row.get('CI_upper', None)
    except Exception:
        # ignore and proceed to other extraction methods
        pass

    # Try to get p-value from clustered GLM result if available
    pval = None
    try:
        res = model_output.get('glm_result_clustered', None) if isinstance(model_output, dict) else None
        if res is not None:
            # res.pvalues is usually a pandas Series
            if hasattr(res, 'pvalues') and 'is_human' in res.pvalues.index:
                pval = res.pvalues['is_human']
                # convert to float if possible
                try:
                    pval = float(pval)
                except Exception:
                    pass
            else:
                # try to compute p from coef and bse if available
                if hasattr(res, 'params') and 'is_human' in res.params.index and hasattr(res, 'bse') and 'is_human' in res.bse.index:
                    coef_tmp = res.params['is_human']
                    bse_tmp = res.bse['is_human']
                    try:
                        z = float(coef_tmp) / float(bse_tmp)
                        # normal approximation for p-value
                        from math import erf, sqrt
                        # cdf = 0.5*(1+erf(z/sqrt(2))) -> two-sided p = 2*(1-cdf_abs)
                        import math as _math
                        from math import sqrt as _sqrt
                        from math import erf as _erf
                        cdf_abs = 0.5 * (1 + _erf(abs(z) / _sqrt(2)))
                        pval = 2 * (1 - cdf_abs)
                    except Exception:
                        pval = None
    except Exception:
        pval = None

    # Attach p-value if found
    if pval is not None:
        try:
            result_obj['p_value'] = float(pval)
        except Exception:
            result_obj['p_value'] = pval

    # Build conclusion logic
    conclusion = None
    note_parts = []

    # If OR is present and finite, interpret it
    OR = result_obj['OR']
    CI_l = result_obj['CI_lower']
    CI_u = result_obj['CI_upper']
    p = result_obj['p_value']

    if OR is not None and is_finite(OR):
        # If p-value available, use it to decide significance
        if p is not None and isinstance(p, (float, int)) and not math.isnan(p):
            if p < 0.05:
                if OR > 1:
                    conclusion = "Yes: modern humans have significantly higher AMTL (OR > 1, p < 0.05)."
                elif OR < 1:
                    conclusion = "No: modern humans have significantly lower AMTL (OR < 1, p < 0.05)."
                else:
                    conclusion = "No difference (OR ~= 1) but p < 0.05 is unexpected; check estimates."
            else:
                conclusion = "No statistically significant difference (p >= 0.05)."
        else:
            # no p-value: use CI if present
            if (CI_l is not None and CI_u is not None) and is_finite(CI_l) and is_finite(CI_u):
                if CI_l > 1:
                    conclusion = "Yes: modern humans have higher AMTL (CI for OR excludes 1, lower bound > 1)."
                elif CI_u < 1:
                    conclusion = "No: modern humans have lower AMTL (CI for OR excludes 1, upper bound < 1)."
                else:
                    conclusion = "Inconclusive: CI for OR includes 1 and no p-value available."
            else:
                conclusion = "Inconclusive: OR present but no finite CI or p-value to assess significance."
    else:
        # OR is missing or infinite -> check for infinite behavior (separation or estimation problems)
        infinite_or = (OR is None) or (not is_finite(OR))
        infinite_ci = (CI_l is None) or (CI_u is None) or (not is_finite(CI_l)) or (not is_finite(CI_u))
        if infinite_or or infinite_ci:
            # If coefficient is extremely large or p-value extremely small, report likely separation
            coef = result_obj['coef']
            try:
                coef_finite = float(coef)
                coef_large = abs(coef_finite) > 1e6
            except Exception:
                coef_finite = None
                coef_large = False

            if coef_large or (p is not None and isinstance(p, (float, int)) and p < 0.001):
                conclusion = ("Apparent strong association: model produced infinite or numerically extreme "
                              "estimates for 'is_human', consistent with complete/near-complete separation. "
                              "This suggests modern humans have much higher AMTL in this dataset, "
                              "but the estimate is not reliable (infinite OR) and indicates model/data issues.")
            else:
                conclusion = ("Estimation problem: OR/CI are infinite or zero (non-finite). "
                              "Unable to draw a reliable statistical conclusion from this model output.")
            note_parts.append("Infinite/zero OR or CI detected — often due to perfect separation, perfect prediction, or problematic weighting.")
        else:
            conclusion = "Inconclusive: insufficient information to determine effect of is_human."

    # Attach note if p-value but large/small
    if result_obj['p_value'] is not None:
        note_parts.append(f"p-value for is_human = {result_obj['p_value']:.4g}")

    # If CI present, add it to note
    if CI_l is not None or CI_u is not None:
        note_parts.append(f"OR CI = [{CI_l}, {CI_u}]")

    # If OR present, add to note
    if OR is not None:
        note_parts.insert(0, f"OR = {OR}")

    result_obj['note'] = "; ".join(note_parts) if note_parts else None
    result_obj['conclusion'] = conclusion

    # Prepare textual description short and targeted to the task question
    if conclusion is None:
        description = ("Could not determine whether modern humans have higher AMTL from the provided model output. "
                       "The model output lacked finite statistics for 'is_human' or a p-value/CI to assess significance.")
    else:
        # Short answer phrased as Yes/No/Inconclusive
        if conclusion.startswith("Yes"):
            short = "Yes"
        elif conclusion.startswith("No"):
            short = "No"
        else:
            short = "Inconclusive"

        description = (f"{short}. {conclusion} "
                       "See 'object' for extracted numeric values and notes about estimation issues (if any).")

    return {
        "object": result_obj,
        "description": description
    }